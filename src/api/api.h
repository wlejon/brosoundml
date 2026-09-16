#pragma once

namespace brosoundml::api {

/// Mounts the 9 audio/voice AI subsystems onto `bro`:
///   - bro.stt (Whisper, FastConformer, Qwen ASR)
///   - bro.tts (Kokoro, Supertonic, OmniVoice, Qwen TTS)
///   - bro.diar (ClusterDiarizer, Sortformer, SpeakerEncoder)
///   - bro.rave (RAVE neural synth)
///   - bro.wake / bro.kws (Wake word and keyword spotting)
///   - bro.sense / bro.gesture / bro.listen (Sensor hub, gesture spotting, listen bus, voice agent)
void installSoundML();

} // namespace brosoundml::api

using brosoundml::api::installSoundML;
