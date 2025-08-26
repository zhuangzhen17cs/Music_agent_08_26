import time
import mido
import app_onnx

def play_in_realtime():
    # 用 app 里加载好的模型
    model = (app_onnx.model_base, app_onnx.model_token, app_onnx.tokenizer)

    # 初始化 prompt（一个 BOS token）
    prompt = [[app_onnx.tokenizer.bos_id] + [app_onnx.tokenizer.pad_id] * (app_onnx.tokenizer.max_token_seq - 1)]
    prompt = np.asarray([prompt], dtype=np.int64)

    # 打开虚拟 MIDI 输出
    with mido.open_output('PythonMIDIOut', virtual=True) as midiout:
        for tokens in app_onnx.generate(model, prompt, batch_size=1, max_len=128):
            token = tokens[0][0]   # 取 batch=1 的第一个 token
            event = app_onnx.tokenizer.tokens2event(token)
            print("event:", event)
            
            # 如果是 note_on/note_off，就发出去
            if event[0] == "note_on":
                midiout.send(mido.Message("note_on", note=event[4], velocity=event[5], channel=event[3]))
            elif event[0] == "note_off":
                midiout.send(mido.Message("note_off", note=event[4], velocity=event[5], channel=event[3]))

            # 简单 sleep，一拍 0.25 秒
            time.sleep(0.25)

if __name__ == "__main__":
    play_in_realtime()
