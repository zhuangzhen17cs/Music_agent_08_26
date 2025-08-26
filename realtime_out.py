# realtime_out.py
import argparse
import time
import threading
import heapq
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np
import mido

import app_onnx  # your module with init_model() and generate()

# ------------------------------------------------------
# --- hard overrides for hardware sanity ---
FORCE_CHANNEL = 0
NOTES_ONLY    = False   # 先关掉过滤，看看模型到底吐了啥
VERBOSE       = True
DEFAULT_GATE_BEATS = 1.0  # 先拉长到 1 拍，容易听见
VEL_FLOOR     = 40



def force_ch(ch: int) -> int:
    return ch if FORCE_CHANNEL < 0 else int(FORCE_CHANNEL)
# ------------------------------------------------------


# ---------------------- Utilities ----------------------

def list_ports():
    outs = mido.get_output_names()
    ins = mido.get_input_names()
    return outs, ins

def pick_port_by_substring(all_names: List[str], substr: str) -> str:
    for n in all_names:
        if substr.lower() in n.lower():
            return n
    raise RuntimeError(f"No MIDI port contains substring: {substr!r}\nAvailable: {all_names}")

# A tiny scheduler to send delayed messages (e.g., note_off) precisely.
class MsgScheduler:
    def __init__(self, midiout):
        self.midiout = midiout
        self.q: List[Tuple[float, mido.Message]] = []
        self.cv = threading.Condition()
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def schedule(self, when_ts: float, msg: mido.Message):
        with self.cv:
            heapq.heappush(self.q, (when_ts, msg))
            self.cv.notify()

    def _loop(self):
        while self.running:
            with self.cv:
                if not self.q:
                    self.cv.wait(timeout=0.05)
                    continue
                when_ts, msg = self.q[0]
                now = time.perf_counter()
                wait = when_ts - now
                if wait > 0:
                    self.cv.wait(timeout=wait)
                    continue
                heapq.heappop(self.q)
            # Send outside the lock
            self.midiout.send(msg)

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify_all()
        self.th.join(timeout=1.0)

# Optional: send MIDI clock as master (24 PPQN)
class ClockSender:
    def __init__(self, midiout, bpm: float):
        self.midiout = midiout
        self.set_bpm(bpm)
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def set_bpm(self, bpm: float):
        # 24 MIDI clocks per quarter note
        self.bpm = max(1.0, float(bpm))
        self.period = 60.0 / (self.bpm * 24.0)

    def _loop(self):
        # Send 'start' to reset transport
        self.midiout.send(mido.Message('start'))
        next_ts = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            if now >= next_ts:
                self.midiout.send(mido.Message('clock'))
                next_ts += self.period
            else:
                time.sleep(min(0.001, next_ts - now))
        # send stop on exit
        self.midiout.send(mido.Message('stop'))

    def stop(self):
        self.running = False
        self.th.join(timeout=1.0)



# ---------------------- Realtime bridge ----------------------

def run_realtime(
    out_port_substr: str,
    bpm: float,
    gate_beats: float,
    send_clock: bool,
    max_events: int,
    temp: float,
    top_p: float,
    top_k: int,
):
    # 1) Init model (CPU/CoreML enforced by your patch)
    model_base, model_token, tokenizer = app_onnx.init_model()
    model = (model_base, model_token, tokenizer)

    # 2) Pick physical output port
    outs, _ = list_ports()
    out_name = pick_port_by_substring(outs, out_port_substr) if out_port_substr else outs[0]
    print(f"[MIDI OUT] -> {out_name}")

    # 3) Open output
    with mido.open_output(out_name) as midiout:
        # 5lines ------------------------------------------------
        midiout.send(mido.Message("program_change", channel=0, program=0))
        midiout.send(mido.Message("note_on", note=60, velocity=110, channel=0))
        time.sleep(0.5)
        midiout.send(mido.Message("note_off", note=60, velocity=0, channel=0))
        print("[SANITY] C4 ping sent on CH1")

        # Optional: be the tempo master by sending MIDI clock
        clock = ClockSender(midiout, bpm) if send_clock else None

        # 4) Create a scheduler to time note_off precisely
        sched = MsgScheduler(midiout)

        # 5) Prepare an initial BOS prompt + seed tempo/program
        seed_events = []
        # set initial tempo
        seed_events.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, int(bpm)]))
        # make sure channel 0 is Acoustic Grand (program 0)
        seed_events.append(tokenizer.event2tokens(["patch_change", 0, 0, 1, 0, 0]))

        prompt_list = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        prompt_list.extend(seed_events)
        prompt = np.asarray([prompt_list], dtype=np.int64)


        # 6) Conversion helpers
        def beats_to_seconds(beats: float) -> float:
            return (60.0 / bpm) * beats

        default_gate_sec = beats_to_seconds(gate_beats)

        # 7) Stream events forever (or until max_events)
        ev_count = 0
        print("[INFO] Streaming tokens... Press Ctrl+C to stop.")
        try:
            # big max_len to keep the generator going
            # gen = app_onnx.generate(
            #     model,
            #     prompt,
            #     batch_size=1,
            #     max_len=10_000_000,
            #     temp=temp,
            #     top_p=top_p,
            #     top_k=top_k,
            #     disable_patch_change=False,
            #     disable_control_change=False,
            #     generator=np.random.RandomState(0),
            # )
            gen = app_onnx.generate(
                model,
                prompt,
                batch_size=1,
                max_len=10_000_000,
                temp=temp,
                top_p=top_p,
                top_k=top_k,
                disable_patch_change=True,
                disable_control_change=True,
                generator=np.random.RandomState(0),
            )
            stats = defaultdict(int)   # 事件类型计数
            dump_first = 100           # 打印前 100 条原始事件
            last_stat_ts = time.time() # 每秒打印一次统计



            for token_seq_batch in gen:
                token_seq = token_seq_batch[0]
                event = tokenizer.tokens2event(token_seq.tolist())
                etype = str(event[0])

                stats[etype] += 1
                if dump_first > 0:
                    print("[EV]", event)
                    dump_first -= 1


                # You may receive time/tempo/program/control/pitch events. We handle common ones:
                if etype == "set_tempo":
                    # event like: ["set_tempo", bar, pos, track, bpm]
                    try:
                        new_bpm = float(event[4])
                        bpm = max(1.0, new_bpm)
                        if clock:
                            clock.set_bpm(bpm)
                        default_gate_sec = beats_to_seconds(gate_beats)
                        print(f"[TEMPO] bpm -> {bpm:.2f}")
                    except Exception:
                        pass

                elif etype in ("patch_change", "program_change"):
                    # ["patch_change", bar, pos, track, channel, program]
                    try:
                        ch = int(event[4]); prog = int(event[5])
                        midiout.send(mido.Message("program_change", program=prog, channel=ch))
                    except Exception:
                        pass

                elif etype == "control_change":
                    # ["control_change", bar, pos, track, channel, control, value]
                    try:
                        ch = int(event[4]); cc = int(event[5]); val = int(event[6])
                        midiout.send(mido.Message("control_change", control=cc, value=val, channel=ch))
                    except Exception:
                        pass

                elif etype == "pitch_wheel":
                    # ["pitch_wheel", bar, pos, track, channel, pitch]
                    try:
                        ch = int(event[4]); pw = int(event[5])
                        midiout.send(mido.Message("pitchwheel", pitch=pw, channel=ch))
                    except Exception:
                        pass

                # elif etype == "note_on":
                #     # Likely: ["note_on", bar, pos, track, channel, note, velocity, ...maybe duration?]
                #     try:
                #         ch = int(event[4]); note = int(event[5]); vel = int(event[6])
                #         midiout.send(mido.Message("note_on", note=note, velocity=vel, channel=ch))

                #         # Try to get duration if present; otherwise use default gate
                #         gate_sec = default_gate_sec
                #         if len(event) >= 8:
                #             # If tokenizer provides duration in beats (best case)
                #             dur_beats = None
                #             # Try common fields; ignore errors silently
                #             for idx in range(7, len(event)):
                #                 val = event[idx]
                #                 if isinstance(val, (int, float)) and 0 < float(val) < 64:
                #                     dur_beats = float(val)
                #                     break
                #             if dur_beats:
                #                 gate_sec = beats_to_seconds(dur_beats)

                #         when_off = time.perf_counter() + gate_sec
                #         off_msg = mido.Message("note_off", note=note, velocity=0, channel=ch)
                #         sched.schedule(when_off, off_msg)
                #     except Exception:
                #         pass

                # elif etype == "note_off":
                #     try:
                #         ch = int(event[4]); note = int(event[5]); vel = int(event[6]) if len(event) > 6 else 0
                #         midiout.send(mido.Message("note_off", note=note, velocity=vel, channel=ch))
                #     except Exception:
                #         pass
                elif etype == "note_on":
                    try:
                        ch = int(event[4] if len(event) > 4 else 0)
                        note = int(event[5] if len(event) > 5 else 60)
                        vel  = int(event[6] if len(event) > 6 else 100)

                        # clamp + force channel
                        ch   = force_ch(max(0, min(15, ch)))
                        note = max(0, min(127, note))
                        vel  = max(VEL_FLOOR, min(127, vel))

                        if VERBOSE:
                            print(f"NOTE_ON  ch={ch} note={note} vel={vel}")

                        midiout.send(mido.Message("note_on", note=note, velocity=vel, channel=ch))

                        gate_sec = (60.0 / bpm) * DEFAULT_GATE_BEATS
                        when_off = time.perf_counter() + gate_sec
                        off_msg  = mido.Message("note_off", note=note, velocity=0, channel=ch)
                        sched.schedule(when_off, off_msg)
                    except Exception as e:
                        print("[ERROR note_on]", e)

                elif etype == "note_off":
                    try:
                        ch = int(event[4] if len(event) > 4 else 0)
                        note = int(event[5] if len(event) > 5 else 60)
                        vel  = int(event[6] if len(event) > 6 else 0)

                        ch   = force_ch(max(0, min(15, ch)))
                        note = max(0, min(127, note))
                        vel  = max(0, min(127, vel))

                        if VERBOSE:
                            print(f"NOTE_OFF ch={ch} note={note} vel={vel}")

                        midiout.send(mido.Message("note_off", note=note, velocity=vel, channel=ch))
                    except Exception as e:
                        print("[ERROR note_off]", e)

                elif etype == "note":
                    # 常见替代格式: ["note", bar, pos, track, channel, pitch, duration_beats, velocity]
                    try:
                        ch   = int(event[4] if len(event) > 4 else 0)
                        note = int(event[5] if len(event) > 5 else 60)
                        durb = float(event[6] if len(event) > 6 else DEFAULT_GATE_BEATS)
                        vel  = int(event[7] if len(event) > 7 else 100)

                        # 合法性 & 钳位
                        ch   = force_ch(max(0, min(15, ch)))
                        note = max(0, min(127, note))
                        vel  = max(VEL_FLOOR, min(127, vel))
                        durb = max(0.05, min(8.0, durb))  # 0.05~8 拍

                        if VERBOSE:
                            print(f"NOTE     ch={ch} note={note} vel={vel} dur={durb} beats")

                        midiout.send(mido.Message("note_on", note=note, velocity=vel, channel=ch))
                        when_off = time.perf_counter() + (60.0 / bpm) * durb
                        sched.schedule(when_off, mido.Message("note_off", note=note, velocity=0, channel=ch))
                    except Exception as e:
                        print("[ERROR note]", e)

                # 现在再做“只发音符”的过滤：非音符直接跳过
                if NOTES_ONLY and etype not in ("note_on", "note_off", "note"):
                    continue

                elif etype == "time_shift":
                    # If tokenizer emits time shifts in "beats" or "steps", sleep accordingly.
                    # We guess a small unit; adjust if you know exact granularity.
                    try:
                        # Try to read a "steps" value and map to 1/64 beat units
                        steps = float(event[-1])
                        sleep_sec = (60.0 / bpm) * (steps / 64.0)
                        if 0.0 < sleep_sec < 2.0:
                            time.sleep(sleep_sec)
                    except Exception:
                        # No usable time info; do nothing
                        pass

                # Optional: throttle a hair to avoid tight loop if there is no timing info
                # Remove if your tokens reliably contain "time_shift"
                # 每秒打印一次事件类型计数
                if time.time() - last_stat_ts > 1.0:
                    print("[STATS]", dict(stats))
                    last_stat_ts = time.time()

                time.sleep(0.0005)

                ev_count += 1
                if 0 < max_events <= ev_count:
                    break

        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted by user.")
        finally:
            if clock:
                clock.stop()
            sched.stop()

# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="", help="Substring of the physical MIDI OUT port name to use")
    parser.add_argument("--bpm", type=float, default=120.0, help="Master tempo if sending clock")
    parser.add_argument("--gate-beats", type=float, default=0.25, help="Default note gate in beats if duration missing")
    parser.add_argument("--send-clock", action="store_true", help="Send MIDI clock (be the master)")
    parser.add_argument("--max-events", type=int, default=0, help="Stop after N events (0 = never)")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.94)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--list-ports", action="store_true", help="List MIDI ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        outs, ins = list_ports()
        print("MIDI OUT ports:")
        for n in outs: print("  -", n)
        print("MIDI IN ports:")
        for n in ins: print("  -", n)
        return

    run_realtime(
        out_port_substr=args.out,
        bpm=args.bpm,
        gate_beats=args.gate_beats,
        send_clock=args.send_clock,
        max_events=args.max_events,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
    )

if __name__ == "__main__":
    main()
