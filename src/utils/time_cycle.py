import re
import torch


def _freq_to_minutes(freq):
    freq = str(freq or "h").lower()
    match = re.match(r"(\d+)", freq)
    mult = int(match.group(1)) if match else 1
    if "min" in freq or freq.endswith("t"):
        return max(mult, 1)
    if freq.endswith("h"):
        return max(mult, 1) * 60
    if freq.endswith("d") or freq.endswith("b"):
        return max(mult, 1) * 60 * 24
    if freq.endswith("w"):
        return max(mult, 1) * 60 * 24 * 7
    if freq.endswith("m"):
        return max(mult, 1) * 60 * 24 * 30
    return 60


def _time_feature_indices(freq):
    freq = str(freq or "h").lower()
    if "min" in freq or freq.endswith("t"):
        return {"minute": 0, "hour": 1, "weekday": 2}
    if freq.endswith("h"):
        return {"hour": 0, "weekday": 1}
    if freq.endswith("d") or freq.endswith("b"):
        return {"weekday": 0}
    return {}


def _denorm_time_feature(values, max_value):
    return torch.clamp(torch.round((values + 0.5) * max_value), 0, max_value).long()


def cycle_index_from_mark(x_mark, cycle, freq="h", timeenc=1, minute_div=15):
    if x_mark is None or cycle is None or cycle <= 0:
        return None
    mark = x_mark
    if mark.dim() < 2:
        return None

    if timeenc == 0:
        if mark.size(-1) < 4:
            return None
        weekday = mark[..., 2].round().long() if mark.size(-1) > 2 else None
        hour = mark[..., 3].round().long()
        if mark.size(-1) > 4:
            minute_bin = mark[..., 4].round().long()
            minute = minute_bin * minute_div
        else:
            minute = torch.zeros_like(hour)
    else:
        hour = torch.zeros(mark.shape[:-1], device=mark.device, dtype=torch.long)
        minute = torch.zeros_like(hour)
        weekday = None
        idx = _time_feature_indices(freq)
        if "hour" in idx and idx["hour"] < mark.size(-1):
            hour = _denorm_time_feature(mark[..., idx["hour"]], 23)
        if "minute" in idx and idx["minute"] < mark.size(-1):
            minute = _denorm_time_feature(mark[..., idx["minute"]], 59)
        if "weekday" in idx and idx["weekday"] < mark.size(-1):
            weekday = _denorm_time_feature(mark[..., idx["weekday"]], 6)

    step_minutes = max(_freq_to_minutes(freq), 1)
    total_minutes = hour * 60 + minute
    step_in_day = total_minutes // step_minutes
    steps_per_day = (24 * 60) // step_minutes
    pos_in_period = step_in_day
    if weekday is not None:
        pos_in_period = weekday * steps_per_day + step_in_day

    return torch.remainder(pos_in_period, cycle).long()
