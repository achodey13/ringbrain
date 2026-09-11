from twilio.twiml.voice_response import Gather, VoiceResponse


def greeting_twiml(business_name: str, gather_action_url: str) -> str:
    """First TwiML served when a call comes in: greet, then listen via Twilio's
    built-in speech recognition (keeps the project's telephony layer simple —
    swap this for a Media Streams + local Whisper pipeline for lower latency
    or on-prem ASR; the agent graph underneath doesn't change either way).
    """
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action=gather_action_url,
        speech_timeout="auto",
        method="POST",
    )
    gather.say(f"Thanks for calling {business_name}. How can I help you today?")
    response.append(gather)
    response.redirect(gather_action_url)  # no speech captured -> re-prompt
    return str(response)


def reply_twiml(reply_text: str, gather_action_url: str, hang_up: bool) -> str:
    response = VoiceResponse()
    if hang_up:
        response.say(reply_text)
        response.hangup()
        return str(response)

    gather = Gather(
        input="speech",
        action=gather_action_url,
        speech_timeout="auto",
        method="POST",
    )
    gather.say(reply_text)
    response.append(gather)
    response.redirect(gather_action_url)
    return str(response)
