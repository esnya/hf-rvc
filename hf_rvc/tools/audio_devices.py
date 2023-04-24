def list_audio_devices(
    input_device: bool = False,
    output_device: bool = False,
):
    """List audio devices."""
    import pyaudio

    pa = pyaudio.PyAudio()

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)

        if input_device and info["maxInputChannels"] == 0:
            continue
        if output_device and info["maxOutputChannels"] == 0:
            continue

        print(pa.get_device_info_by_index(i))

    pa.terminate()


if __name__ == "__main__":
    from argh import ArghParser

    parser = ArghParser()
    parser.set_default_command(list_audio_devices)
    parser.dispatch()
