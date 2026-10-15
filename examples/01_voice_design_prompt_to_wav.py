from qwen3synth import GenerationConfig, Qwen3Runtime, VoiceDesignRequest


def main() -> None:
    with Qwen3Runtime.from_installed(
        "voice-design-1.7b",
        quality="int4",
        provider="cpu",
    ) as runtime:
        result = runtime.design_voice(
            VoiceDesignRequest(
                id="short-prompt-demo",
                text="Hello from Qwen3Synth.",
                description=(
                    "Warm, clear adult narrator, calm pace, natural delivery, "
                    "neutral English accent."
                ),
                language="en",
            ),
            config=GenerationConfig(seed=42, max_new_tokens=64),
        )
        result.save_wav("qwen3_prompt_demo.wav")
        print(
            f"Wrote qwen3_prompt_demo.wav ({result.duration_seconds:.2f}s, {result.sample_rate} Hz)"
        )


if __name__ == "__main__":
    main()
