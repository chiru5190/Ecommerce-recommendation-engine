"""Currency presentation independent of recommendation scoring."""

from decimal import Decimal, ROUND_HALF_UP


def format_inr(value: float | int | Decimal) -> str:
    """Format INR with Indian digit grouping; retain paise only when nonzero."""
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("INR amount must be finite and nonnegative.")
    amount = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    whole, fraction = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        prefix, tail = whole[:-3], whole[-3:]
        groups = []
        while prefix:
            groups.append(prefix[-2:])
            prefix = prefix[:-2]
        whole = ",".join(reversed(groups)) + "," + tail
    return f"₹{whole}" + (f".{fraction}" if fraction != "00" else "")
