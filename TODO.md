# TODO — Deployed App Feedback

Notes from reviewing the deployed app. Ordered easiest/fastest → most complex.

1. ✅ **"ETA at pace" copy is unclear.** Dashboard goal card shows `ETA AT PACE — 8 Feb 2027 — past year-end at current pace`. Both the label and the sub-copy are confusing. Rewrite to something plainer, e.g. label `PROJECTED FINISH` with sub-copy `At current pace you'd hit the goal on this date` (and a separate "behind target" pill if it's after Dec 31).

2. ✅ **Swim dashboard distance units are wrong.** Goal card shows "12 of 78 m" but the number looks like it's in km (not converted to meters). Need sport-aware distance handling — swim should display in m/km, run/bike in km (or mi). Unit label and value must stay in sync.

3. ✅ **Races page: drop year dropdown, show upcoming + past together.** Currently you switch year (2026 shows "Upcoming", 2025 shows "Past Races") — instead, show both sections on one page (e.g. "Upcoming" then "Past Races" below, all-time or grouped by year). Removes the extra click and makes past races discoverable without knowing which year to pick.

4. ✅ **Races Edit form uses the native browser date picker.** It's a light-themed popup that clashes with the dark neon aesthetic everywhere else. Replace with a themed date picker component consistent with the app's styling. Check other forms (Calendar, Goals, Workouts) for the same issue.

5. ✅ **Show gear on the Profile page.** Strava's `/athlete` response already includes `bikes` and `shoes` arrays (name, nickname, total distance, primary flag), and activities carry `gear_id` — so usage stats (count, recent activity) can also be computed locally. Add a "Gear" section to ProfilePage listing shoes + bikes with their totals. Requires surfacing the fields through `get_athlete_profile()` / the athlete router if not already exposed.

6. ✅ **HR zones: let the user pick the source.** Today the app always shows zones estimated from activity data (`/api/athlete/zones` overrides Strava's custom zones even when `custom_zones=True`). On the Profile page, expose three options: (a) use Strava's zones, (b) use app-estimated from activities, (c) enter manually in-app. Persist the choice + manual values, and have analytics/scoring consume whichever source is selected. See `backend/routers/athlete.py:22` and `frontend/src/pages/ProfilePage.tsx:240`.

7. ✅ **Workout Edit form is too complex / needs a rethink.** Each segment exposes Distance, Duration, Fastest pace, Slowest pace, HR Zone, Reps, Recovery time, Recovery dist — many of these are rarely used or redundant. Needs a design pass before coding: trim fields to the common case and align with how workouts are actually used elsewhere (Calendar sessions, matched against activities).

8. ✅ **No mobile support.** On mobile (tested on Android via `strava.rsm-dev.org`), the desktop dock sidebar stays pinned on the left and eats horizontal space, squeezing the main content. Needs a responsive layout — e.g. collapse the dock into a bottom nav or hamburger on small viewports. Touches `AppShell` + every page.
