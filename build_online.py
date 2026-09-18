from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pandas as pd

from work import build_vix20_live_page as dashboard


ROOT = Path(__file__).resolve().parent


def validate_payload(payload: dict, allow_stale: bool = False) -> None:
    coverage = payload["coverage"]
    if len(coverage) != len(dashboard.SYMBOLS):
        raise ValueError("Missing asset coverage")
    dates = {row["last"] for row in coverage}
    if len(dates) != 1:
        raise ValueError(f"Asset data dates differ: {coverage}")
    if payload["dataThrough"] not in dates:
        raise ValueError("Dashboard date does not match input data")
    if not allow_stale:
        latest = pd.Timestamp(payload["dataThrough"])
        today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        if (today - latest).days > 4:
            raise ValueError("Latest data is more than four calendar days old")


ONLINE_UI = """
<style>
@media(max-width:640px){
  .grid2{grid-template-columns:1fr!important}
  .chart{max-width:100%}
  .wrap,.heat{overflow-x:auto}
  #heatmap{min-width:0}
  .heat-row{min-width:620px}
}
</style>
<script>
document.getElementById('updateBtn').textContent='페이지 새로고침';
document.getElementById('updateBtn').onclick=()=>location.reload();
document.getElementById('updateStatus').textContent=
  '매주 토요일 오전 10시(한국시간) 자동 업데이트 예정. 데이터 기준일을 확인하세요.';
const cutoff = new Date(DATA.dataThrough+'T00:00:00Z');
if(Date.now()-cutoff.getTime()>5*86400000){
  const status=document.getElementById('updateStatus');
  status.textContent='주의: 데이터가 오래되었습니다. GitHub Actions의 업데이트 상태를 확인하세요.';
  status.style.color='var(--red)';
}
</script>
"""


def main() -> None:
    allow_stale = os.environ.get("VIX20_OFFLINE_TEST") == "1"
    if os.environ.get("VIX20_USE_CACHE") == "1" and not allow_stale:
        raise ValueError("Online publishing must fetch fresh data, not cache-only data")
    # Publish only after all inputs and the generated payload pass validation.
    for attempt in range(3):
        try:
            payload = dashboard.build_payload()
            validate_payload(payload, allow_stale=allow_stale)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)

    html = dashboard.HTML.replace(
        "__PAYLOAD__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
    )
    html = html.replace("</body>", ONLINE_UI + "</body>")
    public = ROOT / "public"
    public.mkdir(exist_ok=True)
    (public / "index.html").write_text(html, encoding="utf-8")
    (public / ".nojekyll").touch()

    summary = (
        f"Data through: {payload['dataThrough']}\n"
        f"Generated: {payload['generatedAt']}\n"
        f"Monthly signal: {payload['current']['vixSignal']['signalDate']}\n"
        f"Weekly signal: {payload['current']['weeklyFilter']['signalDate']}\n"
        f"Next holding: {payload['current']['nextHolding']}\n"
        f"Adjustment needed: {payload['current']['nextNeedsAdjustment']}\n"
    )
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write("## VIX 2.0 dashboard update\n\n```text\n" + summary + "```\n")


if __name__ == "__main__":
    main()
