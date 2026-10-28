import re

# A tool call the model has opened but not filled yet, at the end of the text:
# "... [Get_weather(Abuja, Nigeria) "
_PENDING_CALL = re.compile(r"\[Get_weather\(([^()]*)\)\s*$")
# A filled tool call: "[Get_weather(Abuja, Nigeria) -> 33.9°C, partly cloudy, wind 3.6 km/h]"
_FILLED_CALL = re.compile(r"\[Get_weather\([^)]*\)\s*->\s*[^\]]*\]")


def parse_pending_call(text: str) -> tuple[str, str] | None:
    match = _PENDING_CALL.search(text)
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",", 1)]
    if len(parts) != 2 or not all(parts):
        return None
    return parts[0], parts[1]


def remove_tool_calls(text: str) -> str:
    return re.sub(r"\s+", " ", _FILLED_CALL.sub("", text)).strip()
