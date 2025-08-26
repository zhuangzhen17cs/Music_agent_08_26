import argparse
import glob
import json
import os.path
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import exit

import gradio as gr
import numpy as np
import onnxruntime as rt
# ===== 最终版 ORT 设备补丁（稳 & 可选 CoreML） =====
# 说明：
# - 默认只用 CPU；若设置环境变量 USE_COREML=1 且可用，则优先 CoreML + CPU。
# - 无论调用方怎么传 device_type/providers，都强制落到我们选的后端。
import os

def _pick_providers():
    avail = set(rt.get_available_providers())
    want = []
    if os.getenv("USE_COREML", "0") == "1" and "CoreMLExecutionProvider" in avail:
        want.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in avail:
        want.append("CPUExecutionProvider")
    if not want:
        # 理论上不会发生；兜底仍返回 CPU
        want = ["CPUExecutionProvider"]
    return want

_FORCE_PROVIDERS = _pick_providers()
_FORCE_DEVICE    = "cpu"  # OrtValue 统一在 CPU 内存上创建，CoreML 也只接受 CPU host tensors

# 1) 强制 InferenceSession 使用我们选定的 providers
_OrigSession = rt.InferenceSession
def _PatchedSession(*args, **kwargs):
    kwargs["providers"] = _FORCE_PROVIDERS
    return _OrigSession(*args, **kwargs)
rt.InferenceSession = _PatchedSession

# 2) OrtValue 工厂函数强制 device_type
_ov = rt.OrtValue

_orig_from_numpy = _ov.ortvalue_from_numpy
def _cpu_from_numpy(*args, **kwargs):
    kwargs["device_type"] = _FORCE_DEVICE
    return _orig_from_numpy(*args, **kwargs)
_ov.ortvalue_from_numpy = staticmethod(_cpu_from_numpy)

_orig_from_shape = _ov.ortvalue_from_shape_and_type
def _cpu_from_shape_and_type(*args, **kwargs):
    kwargs["device_type"] = _FORCE_DEVICE
    return _orig_from_shape(*args, **kwargs)
_ov.ortvalue_from_shape_and_type = staticmethod(_cpu_from_shape_and_type)

# 可选：启动时打印实际 providers，方便你确认没有走到 CUDA 幻觉
print("ONNXRuntime providers ->", _FORCE_PROVIDERS)
# ===== 补丁结束 =====

import requests
import tqdm
from packaging import version

import MIDI
from midi_synthesizer import MidiSynthesizer
from midi_tokenizer import MIDITokenizer

VERSION = "v1.3.5"
# ------------------------------------------
# Globals for external use
model_base = None
model_token = None
tokenizer = None
providers = None
device = None

MAX_SEED = np.iinfo(np.int32).max

rt.set_default_logger_severity(3)


def softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def sample_top_p_k(probs, p, k, generator=None):
    if generator is None:
        generator = np.random
    probs_idx = np.argsort(-probs, axis=-1)
    probs_sort = np.take_along_axis(probs, probs_idx, -1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    mask = np.zeros(probs_sort.shape[-1])
    mask[:k] = 1
    probs_sort = probs_sort * mask
    probs_sort /= np.sum(probs_sort, axis=-1, keepdims=True)
    shape = probs_sort.shape
    probs_sort_flat = probs_sort.reshape(-1, shape[-1])
    probs_idx_flat = probs_idx.reshape(-1, shape[-1])
    next_token = np.stack([generator.choice(idxs, p=pvals) for pvals, idxs in zip(probs_sort_flat, probs_idx_flat)])
    next_token = next_token.reshape(*shape[:-1])
    return next_token

def apply_io_binding(model: rt.InferenceSession, inputs, outputs, batch_size, past_len, cur_len):
    io_binding = model.io_binding()
    for input_ in  model.get_inputs():
        name = input_.name
        if name.startswith("past_key_values"):
            present_name = name.replace("past_key_values", "present")
            if present_name in outputs:
                v = outputs[present_name]
                if past_len > 4095:
                    v = v.numpy()
                    v = v[:, :, -4095:]
                    v = rt.OrtValue.ortvalue_from_numpy(v, device_type=device)
            else:
                v = rt.OrtValue.ortvalue_from_shape_and_type(
                    (batch_size, input_.shape[1], past_len, input_.shape[3]),
                    element_type=np.float32,
                    device_type=device)
            inputs[name] = v
        else:
            v = inputs[name]
        io_binding.bind_ortvalue_input(name, v)

    for output in model.get_outputs():
        name = output.name
        if name.startswith("present"):
            v = rt.OrtValue.ortvalue_from_shape_and_type(
                (batch_size, output.shape[1], cur_len, output.shape[3]),
                element_type=np.float32,
                device_type=device)
            outputs[name] = v
        else:
            v = outputs[name]
        io_binding.bind_ortvalue_output(name, v)
    return io_binding

def generate(model, prompt=None, batch_size=1, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, generator=None):
    tokenizer = model[2]
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    if generator is None:
        generator = np.random
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = np.full((1, max_token_seq), tokenizer.pad_id, dtype=np.int64)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
        input_tensor = input_tensor[None, :, :]
        input_tensor = np.repeat(input_tensor, repeats=batch_size, axis=0)
    else:
        if len(prompt.shape) == 2:
            prompt = prompt[None, :]
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif prompt.shape[0] == 1:
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
            raise ValueError(f"invalid shape for prompt, {prompt.shape}")
        prompt = prompt[..., :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = prompt
    input_tensor = input_tensor[:, -4096:]
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    model0_inputs = {}
    model0_outputs = {}
    emb_size = 1024
    for output in model[0].get_outputs():
        if output.name == "hidden":
            emb_size = output.shape[2]
    past_len = 0
    with bar:
        while cur_len < max_len:
            end = [False] * batch_size
            model0_inputs["x"] = rt.OrtValue.ortvalue_from_numpy(input_tensor[:, past_len:], device_type=device)
            model0_outputs["hidden"] = rt.OrtValue.ortvalue_from_shape_and_type(
                (batch_size, cur_len - past_len, emb_size),
                element_type=np.float32,
                device_type=device)
            io_binding = apply_io_binding(model[0], model0_inputs, model0_outputs, batch_size, past_len, cur_len)
            io_binding.synchronize_inputs()
            model[0].run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            hidden = model0_outputs["hidden"].numpy()[:, -1:]
            next_token_seq = np.zeros((batch_size, 0), dtype=np.int64)
            event_names = [""] * batch_size
            model1_inputs = {"hidden": rt.OrtValue.ortvalue_from_numpy(hidden, device_type=device)}
            model1_outputs = {}
            for i in range(max_token_seq):
                mask = np.zeros((batch_size, tokenizer.vocab_size), dtype=np.int64)
                for b in range(batch_size):
                    if end[b]:
                        mask[b, tokenizer.pad_id] = 1
                        continue
                    if i == 0:
                        mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                        if disable_patch_change:
                            mask_ids.remove(tokenizer.event_ids["patch_change"])
                        if disable_control_change:
                            mask_ids.remove(tokenizer.event_ids["control_change"])
                        mask[b, mask_ids] = 1
                    else:
                        param_names = tokenizer.events[event_names[b]]
                        if i > len(param_names):
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        param_name = param_names[i - 1]
                        mask_ids = tokenizer.parameter_ids[param_name]
                        if param_name == "channel":
                            mask_ids = [i for i in mask_ids if i not in disable_channels]
                        mask[b, mask_ids] = 1
                mask = mask[:, None, :]
                x = next_token_seq
                if i != 0:
                    # cached
                    if i == 1:
                        hidden = np.zeros((batch_size, 0, emb_size), dtype=np.float32)
                        model1_inputs["hidden"] = rt.OrtValue.ortvalue_from_numpy(hidden, device_type=device)
                    x = x[:, -1:]
                model1_inputs["x"] = rt.OrtValue.ortvalue_from_numpy(x, device_type=device)
                model1_outputs["y"] = rt.OrtValue.ortvalue_from_shape_and_type(
                    (batch_size, 1, tokenizer.vocab_size),
                    element_type=np.float32,
                    device_type=device
                )
                io_binding = apply_io_binding(model[1], model1_inputs, model1_outputs, batch_size, i, i+1)
                io_binding.synchronize_inputs()
                model[1].run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()
                logits = model1_outputs["y"].numpy()
                scores = softmax(logits / temp, -1) * mask
                samples = sample_top_p_k(scores, top_p, top_k, generator)
                if i == 0:
                    next_token_seq = samples
                    for b in range(batch_size):
                        if end[b]:
                            continue
                        eid = samples[b].item()
                        if eid == tokenizer.eos_id:
                            end[b] = True
                        else:
                            event_names[b] = tokenizer.id_events[eid]
                else:
                    next_token_seq = np.concatenate([next_token_seq, samples], axis=1)
                    if all([len(tokenizer.events[event_names[b]]) == i for b in range(batch_size) if not end[b]]):
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = np.pad(next_token_seq,
                                        ((0, 0), (0, max_token_seq - next_token_seq.shape[-1])),
                                        mode="constant", constant_values=tokenizer.pad_id)
            next_token_seq = next_token_seq[:, None, :]
            input_tensor = np.concatenate([input_tensor, next_token_seq], axis=1)
            past_len = cur_len
            cur_len += 1
            bar.update(1)
            yield next_token_seq[:, 0]
            if all(end):
                break


def create_msg(name, data):
    return {"name": name, "data": data}


def send_msgs(msgs):
    return json.dumps(msgs)


def run(model_name, tab, mid_seq, continuation_state, continuation_select, instruments, drum_kit, bpm, time_sig,
        key_sig, mid, midi_events, reduce_cc_st, remap_track_channel, add_default_instr, remove_empty_channels,
        seed, seed_rand, gen_events, temp, top_p, top_k, allow_cc):
    global current_model, model_base, model_token, tokenizer
    if current_model != model_name:
        gr.Info("Loading model...")
        model_info = models_info[model_name]
        model_config, model_config_url = model_info[0]
        model_base_path, model_base_url = model_info[1]
        model_token_path, model_token_url = model_info[2]
        try:
            if model_config.endswith(".json"):
                download_if_not_exit(model_config_url, model_config)
            download_if_not_exit(model_base_url, model_base_path)
            download_if_not_exit(model_token_url, model_token_path)
        except Exception as e:
            print(e)
            raise gr.Error("Failed to download files.")
        try:
            model_base = rt.InferenceSession(model_base_path, providers=providers)
            model_token = rt.InferenceSession(model_token_path, providers=providers)
            tokenizer = get_tokenizer(model_config)
            current_model = model_name
            gr.Info("Model loaded")
        except Exception as e:
            print(e)
            raise gr.Error("Failed to load models, maybe you need to delete them and re-download it.")

    bpm = int(bpm)
    if time_sig == "auto":
        time_sig = None
        time_sig_nn = 4
        time_sig_dd = 2
    else:
        time_sig_nn, time_sig_dd = time_sig.split('/')
        time_sig_nn = int(time_sig_nn)
        time_sig_dd = {2: 1, 4: 2, 8: 3}[int(time_sig_dd)]
    if key_sig == 0:
        key_sig = None
        key_sig_sf = 0
        key_sig_mi = 0
    else:
        key_sig = (key_sig - 1)
        key_sig_sf = key_sig // 2 - 7
        key_sig_mi = key_sig % 2
    gen_events = int(gen_events)
    max_len = gen_events
    if seed_rand:
        seed = np.random.randint(0, MAX_SEED)
    generator = np.random.RandomState(seed)
    disable_patch_change = False
    disable_channels = None
    if tab == 0:
        i = 0
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        if tokenizer.version == "v2":
            if time_sig is not None:
                mid.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, time_sig_nn - 1, time_sig_dd - 1]))
            if key_sig is not None:
                mid.append(tokenizer.event2tokens(["key_signature", 0, 0, 0, key_sig_sf + 7, key_sig_mi]))
        if bpm != 0:
            mid.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm]))
        patches = {}
        if instruments is None:
            instruments = []
        for instr in instruments:
            patches[i] = patch2number[instr]
            i = (i + 1) if i != 8 else 10
        if drum_kit != "None":
            patches[9] = drum_kits2number[drum_kit]
        for i, (c, p) in enumerate(patches.items()):
            mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i + 1, c, p]))
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
        if len(instruments) > 0:
            disable_patch_change = True
            disable_channels = [i for i in range(16) if i not in patches]
    elif tab == 1 and mid is not None:
        eps = 4 if reduce_cc_st else 0
        mid = tokenizer.tokenize(MIDI.midi2score(mid), cc_eps=eps, tempo_eps=eps,
                                 remap_track_channel=remap_track_channel,
                                 add_default_instr=add_default_instr,
                                 remove_empty_channels=remove_empty_channels)
        midi_events = int(midi_events)
        if midi_events <= 4096:
            mid = mid[:midi_events]
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
    elif tab == 2 and mid_seq is not None:
        mid = np.asarray(mid_seq, dtype=np.int64)
        if continuation_select > 0:
            continuation_state.append(mid_seq)
            mid = np.repeat(mid[continuation_select - 1:continuation_select], repeats=OUTPUT_BATCH_SIZE, axis=0)
            mid_seq = mid.tolist()
        else:
            continuation_state.append(mid.shape[1])
    else:
        continuation_state = [0]
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()

    if mid is not None:
        max_len += mid.shape[1]

    init_msgs = [create_msg("progress", [0, gen_events])]
    if not (tab == 2 and continuation_select == 0):
        for i in range(OUTPUT_BATCH_SIZE):
            events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
            init_msgs += [create_msg("visualizer_clear", [i, tokenizer.version]),
                          create_msg("visualizer_append", [i, events])]
    yield mid_seq, continuation_state, seed, send_msgs(init_msgs)
    model = (model_base, model_token, tokenizer)
    midi_generator = generate(model, mid, batch_size=OUTPUT_BATCH_SIZE, max_len=max_len, temp=temp,
                              top_p=top_p, top_k=top_k, disable_patch_change=disable_patch_change,
                              disable_control_change=not allow_cc, disable_channels=disable_channels,
                              generator=generator)
    events = [list() for i in range(OUTPUT_BATCH_SIZE)]
    t = time.time()
    for i, token_seqs in enumerate(midi_generator):
        token_seqs = token_seqs.tolist()
        for j in range(OUTPUT_BATCH_SIZE):
            token_seq = token_seqs[j]
            mid_seq[j].append(token_seq)
            events[j].append(tokenizer.tokens2event(token_seq))
        if time.time() - t > 0.2:
            msgs = [create_msg("progress", [i + 1, gen_events])]
            for j in range(OUTPUT_BATCH_SIZE):
                msgs += [create_msg("visualizer_append", [j, events[j]])]
                events[j] = list()
            yield mid_seq, continuation_state, seed, send_msgs(msgs)
            t = time.time()
    yield mid_seq, continuation_state, seed, send_msgs([])


def finish_run(mid_seq):
    if mid_seq is None:
        outputs = [None] * OUTPUT_BATCH_SIZE
        return *outputs, []
    outputs = []
    end_msgs = [create_msg("progress", [0, 0])]
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    for i in range(OUTPUT_BATCH_SIZE):
        events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
        mid = tokenizer.detokenize(mid_seq[i])
        with open(f"outputs/output{i + 1}.mid", 'wb') as f:
            f.write(MIDI.score2midi(mid))
        outputs.append(f"outputs/output{i + 1}.mid")
        end_msgs += [create_msg("visualizer_clear", [i, tokenizer.version]),
                     create_msg("visualizer_append", [i, events]),
                     create_msg("visualizer_end", i)]
    return *outputs, send_msgs(end_msgs)


def synthesis_task(mid):
    return synthesizer.synthesis(MIDI.score2opus(mid))

def render_audio(mid_seq, should_render_audio):
    if (not should_render_audio) or mid_seq is None:
        outputs = [None] * OUTPUT_BATCH_SIZE
        return tuple(outputs)
    outputs = []
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    audio_futures = []
    for i in range(OUTPUT_BATCH_SIZE):
        mid = tokenizer.detokenize(mid_seq[i])
        audio_future = thread_pool.submit(synthesis_task, mid)
        audio_futures.append(audio_future)
    for future in audio_futures:
        outputs.append((44100, future.result()))
    if OUTPUT_BATCH_SIZE == 1:
        return outputs[0]
    return tuple(outputs)


def undo_continuation(mid_seq, continuation_state):
    if mid_seq is None or len(continuation_state) < 2:
        return mid_seq, continuation_state, send_msgs([])
    if isinstance(continuation_state[-1], list):
        mid_seq = continuation_state[-1]
    else:
        mid_seq = [ms[:continuation_state[-1]] for ms in mid_seq]
    continuation_state = continuation_state[:-1]
    end_msgs = [create_msg("progress", [0, 0])]
    for i in range(OUTPUT_BATCH_SIZE):
        events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
        end_msgs += [create_msg("visualizer_clear", [i, tokenizer.version]),
                     create_msg("visualizer_append", [i, events]),
                     create_msg("visualizer_end", i)]
    return mid_seq, continuation_state, send_msgs(end_msgs)


def download(url, output_file):
    print(f"Downloading {output_file} from {url}")
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    with tqdm.tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                   desc=f"Downloading {output_file}") as pbar:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_if_not_exit(url, output_file):
    if os.path.exists(output_file):
        return
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        download(url, output_file)
    except Exception as e:
        print(f"Failed to download {output_file} from {url}")
        raise e


def load_javascript(dir="javascript"):
    scripts_list = glob.glob(f"{dir}/*.js")
    javascript = ""
    for path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            js_content = jsfile.read()
            js_content = js_content.replace("const MIDI_OUTPUT_BATCH_SIZE=4;",
                                            f"const MIDI_OUTPUT_BATCH_SIZE={OUTPUT_BATCH_SIZE};")
            javascript += f"\n<!-- {path} --><script>{js_content}</script>"
    template_response_ori = gr.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = template_response_ori(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response

def get_tokenizer(config_name_or_path):
    if config_name_or_path.endswith(".json"):
        with open(config_name_or_path, "r") as f:
            config = json.load(f)
        tv = config["tokenizer"]["version"]
        o = config["tokenizer"]["optimise_midi"]
    else:
        tv, size = config_name_or_path.split("-")
        tv = tv[1:]
        if tv[-1] == "o":
            o = True
            tv = tv[:-1]
        else:
            o = False
        if tv not in ["v1", "v2"]:
            raise ValueError(f"Unknown tokenizer version {tv}")
    tokenizer = MIDITokenizer(tv)
    tokenizer.set_optimise_midi(o)
    return tokenizer


def check_update(current_ver):
    v1 = version.parse(current_ver)
    url = f"https://api.github.com/repos/SkyTNT/midi-model/releases/latest"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            latest_release = response.json()
            v2 = version.parse(latest_release['tag_name'])
            if v2 > v1:
                print(f"A new version (v{v2}) is available! You are currently using v{v1}."
                      f" Please update to the latest version for new features and improvements."
                      f" https://github.com/SkyTNT/midi-model/releases")
    except Exception:
        print("Failed to Check update")

number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}
key_signatures = ['C♭', 'A♭m', 'G♭', 'E♭m', 'D♭', 'B♭m', 'A♭', 'Fm', 'E♭', 'Cm', 'B♭', 'Gm', 'F', 'Dm',
                  'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F♯m', 'E', 'C♯m', 'B', 'G♯m', 'F♯', 'D♯m', 'C♯', 'A♯m']

# ------------------------------------------
def init_model(
    model_config_path="models/default/config.json",
    model_base_path="models/default/model_base.onnx",
    model_token_path="models/default/model_token.onnx",
    model_config_url="https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/config.json",
    model_base_url="https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/onnx/model_base.onnx",
    model_token_url="https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/onnx/model_token.onnx",
):
    """
    Initialize and return (model_base, model_token, tokenizer) for external realtime scripts.
    Does not launch Gradio. Uses the same provider patch (CPU/CoreML) you already applied.
    """
    global model_base, model_token, tokenizer, providers, device

    # Download artifacts if missing (same logic as your main)
    if model_config_path.endswith(".json"):
        download_if_not_exit(model_config_url, model_config_path)
    download_if_not_exit(model_base_url, model_base_path)
    download_if_not_exit(model_token_url, model_token_path)

    # Providers/device follow your top-level ORT patch (forced to CPU or CoreML+CPU)
    providers = _FORCE_PROVIDERS
    device = _FORCE_DEVICE  # used by your OrtValue factory; patched to CPU anyway

    # Create tokenizer and sessions
    tokenizer = get_tokenizer(model_config_path)
    model_base = rt.InferenceSession(model_base_path, providers=providers)
    model_token = rt.InferenceSession(model_token_path, providers=providers)

    return model_base, model_token, tokenizer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--port", type=int, default=-1, help="gradio server port")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--max-gen", type=int, default=4096, help="max")
    parser.add_argument("--soundfont-path", type=str, default="soundfont.sf2", help="soundfont")
    parser.add_argument("--model-config", type=str,
                        default="models/default/config.json",
                        help="model config name or path")
    parser.add_argument("--model-base-path", type=str,
                        default="models/default/model_base.onnx", help="model path")
    parser.add_argument("--model-token-path", type=str,
                        default="models/default/model_token.onnx", help="model path")
    parser.add_argument("--soundfont-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model/resolve/main/soundfont.sf2",
                        help="download soundfont to soundfont-path if file not exist")
    parser.add_argument("--model-config-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/config.json",
                        help="download config.json to model-config if file not exist")
    parser.add_argument("--model-base-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/onnx/model_base.onnx",
                        help="download model-base to model-base-path if file not exist")
    parser.add_argument("--model-token-url", type=str,
                        default="https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/onnx/model_token.onnx",
                        help="download model-token to model-token-path if file not exist")
    opt = parser.parse_args()
    check_update(VERSION)
    OUTPUT_BATCH_SIZE = opt.batch
    models_info = {
        "generic pretrain model (tv2o-medium) by skytnt (default)": [
            [opt.model_config, opt.model_config_url],
            [opt.model_base_path, opt.model_base_url],
            [opt.model_token_path, opt.model_token_url]
        ],
        "generic pretrain model (tv2o-medium) by skytnt with jpop lora": [
            ["models/tv2om_skytnt_jpop_lora/config.json",
             "https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/config.json"],
            ["models/tv2om_skytnt_jpop_lora/model_base.onnx",
             "https://huggingface.co/skytnt/midi-model-tv2om-jpop-lora/resolve/main/onnx/model_base.onnx"],
            ["models/tv2om_skytnt_jpop_lora/model_token.onnx",
             "https://huggingface.co/skytnt/midi-model-tv2om-jpop-lora/resolve/main/onnx/model_token.onnx"]
        ],
        "generic pretrain model (tv2o-medium) by skytnt with touhou lora": [
            ["models/tv2om_skytnt_touhou_lora/config.json",
             "https://huggingface.co/skytnt/midi-model-tv2o-medium/resolve/main/config.json"],
            ["models/tv2om_skytnt_touhou_lora/model_base.onnx",
             "https://huggingface.co/skytnt/midi-model-tv2om-touhou-lora/resolve/main/onnx/model_base.onnx"],
            ["models/tv2om_skytnt_touhou_lora/model_token.onnx",
             "https://huggingface.co/skytnt/midi-model-tv2om-touhou-lora/resolve/main/onnx/model_token.onnx"]
        ],
        "generic pretrain model (tv2o-large) by asigalov61": [
            ["models/tv2ol_asigalov61/config.json",
             "https://huggingface.co/asigalov61/Music-Llama/resolve/main/config.json"],
            ["models/tv2ol_asigalov61/model_base.onnx",
             "https://huggingface.co/asigalov61/Music-Llama/resolve/main/onnx/model_base.onnx"],
            ["models/tv2ol_asigalov61/model_token.onnx",
             "https://huggingface.co/asigalov61/Music-Llama/resolve/main/onnx/model_token.onnx"]
        ],
        "generic pretrain model (tv2o-medium) by asigalov61": [
            ["models/tv2om_asigalov61/config.json",
             "https://huggingface.co/asigalov61/Music-Llama-Medium/resolve/main/config.json"],
            ["models/tv2om_asigalov61/model_base.onnx",
             "https://huggingface.co/asigalov61/Music-Llama-Medium/resolve/main/onnx/model_base.onnx"],
            ["models/tv2om_asigalov61/model_token.onnx",
             "https://huggingface.co/asigalov61/Music-Llama-Medium/resolve/main/onnx/model_token.onnx"]
        ],
        "generic pretrain model (tv1-medium) by skytnt": [
            ["models/tv1m_skytnt/config.json",
             "https://huggingface.co/skytnt/midi-model/resolve/main/config.json"],
            ["models/tv1m_skytnt/model_base.onnx",
             "https://huggingface.co/skytnt/midi-model/resolve/main/onnx/model_base.onnx"],
            ["models/tv1m_skytnt/model_token.onnx",
             "https://huggingface.co/skytnt/midi-model/resolve/main/onnx/model_token.onnx"]
        ]
    }
    current_model = list(models_info.keys())[0]
    try:
        download_if_not_exit(opt.soundfont_url, opt.soundfont_path)
        if opt.model_config.endswith(".json"):
            download_if_not_exit(opt.model_config_url, opt.model_config)
        download_if_not_exit(opt.model_base_url, opt.model_base_path)
        download_if_not_exit(opt.model_token_url, opt.model_token_path)
    except Exception as e:
        print(e)
        input("Failed to download files.\nPress any key to continue...")
        exit(-1)
    soundfont_path = opt.soundfont_path
    synthesizer = MidiSynthesizer(soundfont_path)
    thread_pool = ThreadPoolExecutor(max_workers=OUTPUT_BATCH_SIZE)
    tokenizer = get_tokenizer(opt.model_config)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    device = "cuda"
    try:
        model_base = rt.InferenceSession(opt.model_base_path, providers=providers)
        model_token = rt.InferenceSession(opt.model_token_path, providers=providers)
    except Exception as e:
        print(e)
        input("Failed to load models, maybe you need to delete them and re-download it.\nPress any key to continue...")
        exit(-1)

    load_javascript()
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Midi Composer</h1>")
        # gr.Markdown("![Visitors](https://api.visitorbadge.io/api/visitors?path=skytnt.midi-composer&style=flat)\n\n"
        #             "Midi event transformer for music generation\n\n"
        #             "Demo for [SkyTNT/midi-model](https://github.com/SkyTNT/midi-model)\n\n"
        #             "[Open In Colab]"
        #             "(https://colab.research.google.com/github/SkyTNT/midi-model/blob/main/demo.ipynb)"
        #             " for faster running and longer generation\n\n"
        #             "**Update v1.3**: MIDITokenizerV2 and new MidiVisualizer\n\n"
        #             "The current **best** model: generic pretrain model (tv2o-medium) by skytnt"
        #             )
        js_msg = gr.Textbox(elem_id="msg_receiver", visible=False)
        js_msg.change(None, [js_msg], [], js="""
        (msg_json) =>{
            let msgs = JSON.parse(msg_json);
            executeCallbacks(msgReceiveCallbacks, msgs);
            return [];
        }
        """)
        input_model = gr.Dropdown(label="select model", choices=list(models_info.keys()),
                                  type="value", value=list(models_info.keys())[0])
        tab_select = gr.State(value=0)
        with gr.Tabs():
            with gr.TabItem("custom prompt") as tab1:
                input_instruments = gr.Dropdown(label="🪗instruments (auto if empty)", choices=list(patch2number.keys()),
                                                multiselect=True, max_choices=15, type="value")
                input_drum_kit = gr.Dropdown(label="🥁drum kit", choices=list(drum_kits2number.keys()), type="value",
                                             value="None")
                input_bpm = gr.Slider(label="BPM (beats per minute, auto if 0)", minimum=0, maximum=255,
                                              step=1,
                                              value=0)
                input_time_sig = gr.Radio(label="time signature (only for tv2 models)",
                                             value="auto",
                                             choices=["auto", "4/4", "2/4", "3/4", "6/4", "7/4",
                                                      "2/2", "3/2", "4/2", "3/8", "5/8", "6/8", "7/8", "9/8", "12/8"]
                                             )
                input_key_sig = gr.Radio(label="key signature (only for tv2 models)",
                                            value="auto",
                                            choices=["auto"] + key_signatures,
                                            type="index"
                                            )
                example1 = gr.Examples([
                    [[], "None"],
                    [["Acoustic Grand"], "None"],
                    [['Acoustic Grand', 'SynthStrings 2', 'SynthStrings 1', 'Pizzicato Strings',
                      'Pad 2 (warm)', 'Tremolo Strings', 'String Ensemble 1'], "Orchestra"],
                    [['Trumpet', 'Oboe', 'Trombone', 'String Ensemble 1', 'Clarinet',
                      'French Horn', 'Pad 4 (choir)', 'Bassoon', 'Flute'], "None"],
                    [['Flute', 'French Horn', 'Clarinet', 'String Ensemble 2', 'English Horn', 'Bassoon',
                      'Oboe', 'Pizzicato Strings'], "Orchestra"],
                    [['Electric Piano 2', 'Lead 5 (charang)', 'Electric Bass(pick)', 'Lead 2 (sawtooth)',
                      'Pad 1 (new age)', 'Orchestra Hit', 'Cello', 'Electric Guitar(clean)'], "Standard"],
                    [["Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"]
                ], [input_instruments, input_drum_kit])
            with gr.TabItem("midi prompt") as tab2:
                input_midi = gr.File(label="input midi", file_types=[".midi", ".mid"], type="binary")
                input_midi_events = gr.Slider(label="use first n midi events as prompt (all if 4097)", minimum=1,
                                              maximum=4097,
                                              step=1,
                                              value=128)
                input_reduce_cc_st = gr.Checkbox(label="reduce control_change and set_tempo events", value=True)
                input_remap_track_channel = gr.Checkbox(
                    label="remap tracks and channels so each track has only one channel and in order", value=True)
                input_add_default_instr = gr.Checkbox(
                    label="add a default instrument to channels that don't have an instrument", value=True)
                input_remove_empty_channels = gr.Checkbox(label="remove channels without notes", value=False)
            with gr.TabItem("last output prompt") as tab3:
                gr.Markdown("Continue generating on the last output.")
                input_continuation_select = gr.Radio(label="select output to continue generating", value="all",
                                                     choices=["all"] + [f"output{i + 1}" for i in
                                                                        range(OUTPUT_BATCH_SIZE)],
                                                     type="index"
                                                     )
                undo_btn = gr.Button("undo the last continuation")

        tab1.select(lambda: 0, None, tab_select, queue=False)
        tab2.select(lambda: 1, None, tab_select, queue=False)
        tab3.select(lambda: 2, None, tab_select, queue=False)
        input_seed = gr.Slider(label="seed", minimum=0, maximum=2 ** 31 - 1,
                               step=1, value=0)
        input_seed_rand = gr.Checkbox(label="random seed", value=True)
        input_gen_events = gr.Slider(label="generate max n midi events", minimum=1, maximum=opt.max_gen,
                                     step=1, value=opt.max_gen // 2)
        with gr.Accordion("options", open=False):
            input_temp = gr.Slider(label="temperature", minimum=0.1, maximum=1.2, step=0.01, value=1)
            input_top_p = gr.Slider(label="top p", minimum=0.1, maximum=1, step=0.01, value=0.94)
            input_top_k = gr.Slider(label="top k", minimum=1, maximum=128, step=1, value=20)
            input_allow_cc = gr.Checkbox(label="allow midi cc event", value=True)
            # change true to false for faster generation
            input_render_audio = gr.Checkbox(label="render audio after generation", value=True)
            example3 = gr.Examples([[1, 0.94, 128], [1, 0.98, 20], [1, 0.98, 12]],
                                   [input_temp, input_top_p, input_top_k])
        run_btn = gr.Button("generate", variant="primary")
        stop_btn = gr.Button("stop and output")
        output_midi_seq = gr.State()
        output_continuation_state = gr.State([0])
        midi_outputs = []
        audio_outputs = []
        with gr.Tabs(elem_id="output_tabs"):
            for i in range(OUTPUT_BATCH_SIZE):
                with gr.TabItem(f"output {i + 1}") as tab1:
                    output_midi_visualizer = gr.HTML(elem_id=f"midi_visualizer_container_{i}")
                    output_audio = gr.Audio(label="output audio", format="wav", elem_id=f"midi_audio_{i}")
                    output_midi = gr.File(label="output midi", file_types=[".mid"])
                    midi_outputs.append(output_midi)
                    audio_outputs.append(output_audio)
        run_event = run_btn.click(run, [input_model, tab_select, output_midi_seq, output_continuation_state,
                                        input_continuation_select, input_instruments, input_drum_kit, input_bpm,
                                        input_time_sig, input_key_sig, input_midi, input_midi_events,
                                        input_reduce_cc_st, input_remap_track_channel, input_add_default_instr,
                                        input_remove_empty_channels, input_seed, input_seed_rand, input_gen_events,
                                        input_temp, input_top_p, input_top_k, input_allow_cc],
                                  [output_midi_seq, output_continuation_state, input_seed, js_msg],
                                  concurrency_limit=3, queue=True)
        finish_run_event = run_event.then(fn=finish_run,
                                          inputs=[output_midi_seq],
                                          outputs=midi_outputs + [js_msg],
                                          queue=False)
        finish_run_event.then(fn=render_audio,
                              inputs=[output_midi_seq, input_render_audio],
                              outputs=audio_outputs,
                              queue=False)
        stop_btn.click(None, [], [], cancels=run_event, queue=False)
        undo_btn.click(undo_continuation, [output_midi_seq, output_continuation_state],
                            [output_midi_seq, output_continuation_state, js_msg], queue=False)
    try:
        port = opt.port
        if port == -1:
            port = None
        # load_javascript not work on ssr mode
        app.launch(server_port=port, share=opt.share, inbrowser=True, ssr_mode=False)
    except Exception as e:
        print(e)
        input("Failed to launch webui.\nPress any key to continue...")
        exit(-1)
    finally:
        thread_pool.shutdown()


# print("providers:", sess0.get_providers())