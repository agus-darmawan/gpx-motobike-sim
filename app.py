import os
import io
import uuid
import math
import random
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import gpxpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from werkzeug.utils import secure_filename

# ---------- CONFIG ----------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {"gpx", "GPX"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
# ----------------------------

# ----------------------------
# Utilities
# ----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    # meters
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

# ----------------------------
# GPX loader
# ----------------------------
def load_gpx_trackpoints(path):
    with open(path, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)
    pts = []
    # prefer track points
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                ele = p.elevation if p.elevation is not None else 0.0
                pts.append((p.latitude, p.longitude, ele))
    if len(pts) == 0:
        for w in gpx.waypoints:
            ele = w.elevation if w.elevation is not None else 0.0
            pts.append((w.latitude, w.longitude, ele))
    return pts

# ----------------------------
# Event placement helpers
# ----------------------------
def place_events_along_route(total_dist_m, seed=0):
    random.seed(seed)
    events = []  # (distance_m, event_name)
    # lampu merah: every 1-3 km randomly (simulate intersections)
    pos = 0
    while pos < total_dist_m:
        step = random.uniform(500, 3000)
        pos += step
        if pos < total_dist_m and random.random() < 0.4:  # not every intersection has red in route
            events.append((pos, "lampu_merah"))
    # polisi tidur: many small bumps
    pos = 0
    while pos < total_dist_m:
        pos += random.uniform(200, 900)
        if pos < total_dist_m and random.random() < 0.6:
            events.append((pos, "polisi_tidur"))
    # macet pockets
    for _ in range(int(max(1, total_dist_m // 5000))):
        pos = random.uniform(500, max(1000, total_dist_m-500))
        events.append((pos, "macet"))
    # random sudden braking events
    for _ in range(int(total_dist_m // 20000) + 1):
        pos = random.uniform(0, total_dist_m)
        events.append((pos, "sudden_brake"))
    # normalize sort
    events_sorted = sorted(events, key=lambda x: x[0])
    return events_sorted

# ----------------------------
# Simulation core (motor 4-stroke tuned)
# ----------------------------
def simulate_motor_4t(track_points, seed=42):
    """
    Input: list of (lat, lon, ele)
    Output: pandas.DataFrame per-second rows with:
      time_s, lat, lon, ele_m, speed_kmh, dist_m, brake (bool), event (str)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Parameters tuned for motor 4T:
    base_cruise_kmh = 55.0  # typical comfortable average
    max_speed_kmh = 120.0
    min_speed_kmh = 0.0
    max_acc_m_s2 = 2.6   # motorcycle can accelerate fairly briskly ~2-3 m/s2
    max_dec_m_s2 = 6.5   # emergency ~6 m/s2
    brake_threshold = 1.2 # m/s2 threshold to mark brake

    # Build segment list
    n = len(track_points)
    if n < 2:
        raise ValueError("GPX must have at least 2 points")

    segs = []
    total = 0.0
    for i in range(n-1):
        lat1, lon1, e1 = track_points[i]
        lat2, lon2, e2 = track_points[i+1]
        d = haversine_m(lat1, lon1, lat2, lon2)
        total += d
        slope = (e2 - e1) / d if d > 0 else 0.0
        segs.append({
            'i': i,
            'lat1': lat1, 'lon1': lon1, 'ele1': e1,
            'lat2': lat2, 'lon2': lon2, 'ele2': e2,
            'dist': d,
            'slope': slope
        })

    events = place_events_along_route(total, seed=seed)

    # Map events to segments by distance thresholds
    cum = [0.0]
    for s in segs:
        cum.append(cum[-1] + s['dist'])

    seg_event = [None] * len(segs)
    for ev_pos, ev_name in events:
        # find segment index covering this distance
        for j in range(len(segs)):
            if cum[j] <= ev_pos <= cum[j+1]:
                # if multiple events map, combine into list string
                if seg_event[j] is None:
                    seg_event[j] = ev_name
                else:
                    seg_event[j] = seg_event[j] + "|" + ev_name
                break

    # We'll produce a coarse per-segment simulation first:
    records = []  # (time_s, lat, lon, ele, speed_kmh, dist_m, brake, event)
    time_s = 0.0
    dist_accum = 0.0
    speed = base_cruise_kmh / 3.6  # start m/s
    last_speed = speed

    # small helper to add jitter / realistic driver behavior
    def random_drive_factor():
        # returns multiplier around 0.9..1.2 with occasional bursts
        r = random.random()
        if r < 0.02:
            return random.uniform(1.2, 1.6)  # short burst / overtaking
        if r < 0.10:
            return random.uniform(0.7, 0.95)  # cautious
        return random.uniform(0.95, 1.08)

    for idx, s in enumerate(segs):
        # baseline target speed from geometry & slope
        slope = s['slope']
        # slope effect: uphill reduce, downhill slight increase
        if slope > 0.03:
            slope_penalty = 0.20 + min(0.6, slope * 10)
        elif slope < -0.04:
            slope_penalty = -0.08  # small boost downhill
        else:
            slope_penalty = 0.0

        # curvature approx via three points if available (to penalize turns)
        angle_rad = 0.0
        i = s['i']
        if 0 < i < len(track_points) - 1:
            a = (track_points[i-1][0], track_points[i-1][1])
            b = (track_points[i][0], track_points[i][1])
            c = (track_points[i+1][0], track_points[i+1][1])
            # small vector method
            def vec(p, q):
                lat1, lon1 = p; lat2, lon2 = q
                x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1+lat2)/2)) * 6371000
                y = math.radians(lat2 - lat1) * 6371000
                return (x, y)
            v1 = vec(b, a); v2 = vec(b, c)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            m1 = math.hypot(v1[0], v1[1]); m2 = math.hypot(v2[0], v2[1])
            if m1*m2 > 0:
                cosang = max(-1.0, min(1.0, dot/(m1*m2)))
                angle_rad = math.acos(cosang)

        curvature_penalty = (angle_rad / math.radians(90)) * 0.95  # up to ~0.95 multiplier effect

        # base target km/h
        target_kmh = base_cruise_kmh * (1.0 - 0.6 * curvature_penalty - slope_penalty)
        # apply random drive factor
        target_kmh *= random_drive_factor()
        target_kmh = max(10.0, min(max_speed_kmh, target_kmh))

        # apply events on this seg
        event = seg_event[idx]
        if event is not None:
            # handle combined events if '|' present: check priority
            ev_list = event.split("|")
            if "lampu_merah" in ev_list:
                # full stop on this seg near its end, but keep some approach braking
                # We'll set target temporarily to very low (0-3 km/h)
                target_kmh = random.uniform(0.0, 2.0)
            if "polisi_tidur" in ev_list:
                # speed bump: slow to 6-20 km/h depending on randomness
                bump_speed = random.uniform(6.0, 18.0)
                target_kmh = min(target_kmh, bump_speed)
            if "macet" in ev_list:
                target_kmh = random.uniform(3.0, 18.0)
            if "sudden_brake" in ev_list:
                # simulate an abrupt braking then recovery
                target_kmh = max(0.0, speed*3.6 - random.uniform(10, 40))

        # If long straight with small curvature and small slope, sometimes allow burst (ngebut)
        if curvature_penalty < 0.05 and abs(slope) < 0.02 and random.random() < 0.03:
            target_kmh = min(max_speed_kmh, target_kmh * random.uniform(1.2, 1.6))

        # Now convert to m/s
        target = target_kmh / 3.6
        segdist = s['dist']

        # Integrate across the segment but we will record per-second values later using interpolation;
        # For now compute a reasonable traversal time given acceleration limits but allow variability.
        # We'll attempt to accelerate/decelerate towards target with bounded accel.
        # Compute an estimate of new_speed after traversing this segment
        # Use simplistic physics: v^2 = u^2 + 2*a*s  with limited a toward target
        u = speed
        desired = target

        # Choose accel sign
        if desired > u:
            # accelerate, but use random fraction of max_acc
            a = max_acc_m_s2 * random.uniform(0.5, 1.0)
            # limit obtainable speed by v = sqrt(u^2 + 2as)
            v_lim = math.sqrt(max(0.0, u*u + 2*a*segdist))
            v_end = min(v_lim, desired)
        else:
            # decelerate (brake) possibly stronger
            a = max_dec_m_s2 * random.uniform(0.4, 1.0)
            v_sq = max(0.0, u*u - 2*a*segdist)
            v_end = math.sqrt(v_sq)
            # sometimes emergency brake (sudden) if event present
            if event is not None and ("sudden_brake" in event or "lampu_merah" in event):
                # Force quick stop
                v_end = min(v_end, desired, u * random.uniform(0.0, 0.6))

        # estimate time to traverse using trapezoidal average speed
        avg_v = max(0.01, (u + v_end) / 2.0)
        dt = segdist / avg_v if avg_v > 0 else 1.0
        # but enforce at least small integer seconds for per-second sampling
        # We'll create per-second samples by interpolation; compute number of seconds ~ round(dt)
        nsec = max(1, int(round(dt)))
        # Now produce per-second interpolation across this segment
        for sidx in range(nsec):
            frac = (sidx + 1) / nsec
            # position interpolate
            lat = s['lat1'] + (s['lat2'] - s['lat1']) * frac
            lon = s['lon1'] + (s['lon2'] - s['lon1']) * frac
            ele = s['ele1'] + (s['ele2'] - s['ele1']) * frac
            # speed interpolation with small micro-jitter to avoid overly smooth
            # use easing curve and noise
            sp = (u + (v_end - u) * frac)
            # add micro random variation (simulating throttle variability), but avoid negative
            micro = random.normalvariate(0, 0.3)  # m/s noise
            sp = max(0.0, sp + micro)
            # braking detection: if instantaneous accel < -brake_threshold mark brake True
            # We'll estimate acceleration by derivative across this second vs previous
            # but as we don't have previous per-second yet, we'll approximate later.
            records.append({
                'time_s': time_s,
                'lat': lat,
                'lon': lon,
                'ele_m': ele,
                'speed_m_s': sp,
                'dist_m': dist_accum + segdist * frac,
                'event': event
            })
            time_s += 1.0  # one sample per second
        # update state for next segment
        speed = v_end
        dist_accum += segdist

    # finish: convert records to DataFrame, compute speed_kmh, brake flag by accel threshold
    df = pd.DataFrame(records)
    # compute accelerations (m/s2) using diff of speed_m_s over time
    df['speed_kmh'] = df['speed_m_s'] * 3.6
    df['time_s'] = df['time_s'].astype(int)  # integer seconds
    df = df[['time_s','lat','lon','ele_m','speed_kmh','dist_m','event','speed_m_s']]

    # compute accel using forward difference
    df = df.reset_index(drop=True)
    df['accel_m_s2'] = 0.0
    for i in range(1, len(df)):
        dv = df.loc[i, 'speed_m_s'] - df.loc[i-1, 'speed_m_s']
        dt = max(1.0, df.loc[i, 'time_s'] - df.loc[i-1, 'time_s'])
        df.loc[i, 'accel_m_s2'] = dv / dt

    # brake flag: accel less than -brake_threshold OR event includes lampu_merah/ sudden_brake
    def detect_brake(row):
        if row['accel_m_s2'] < -brake_threshold:
            return True
        ev = row['event']
        if isinstance(ev, str) and ('lampu_merah' in ev or 'sudden_brake' in ev):
            # but only if speed is low or decreasing
            return True
        return False

    df['brake'] = df.apply(detect_brake, axis=1)
    # cleanup columns and order
    df_out = df[['time_s','lat','lon','ele_m','speed_kmh','dist_m','brake','event']].copy()
    # ensure types
    df_out['time_s'] = df_out['time_s'].astype(int)
    df_out['brake'] = df_out['brake'].astype(bool)
    # Remove possible duplicate time rows (could appear if two segments have same time); keep first per time
    df_out = df_out.groupby('time_s', as_index=False).first()
    return df_out

# ----------------------------
# Plotting utility
# ----------------------------
def make_plots(df, out_prefix):
    # produce three plots, save to files
    t = df['time_s'].values
    speed = df['speed_kmh'].values
    dist = df['dist_m'].values
    ele = df['ele_m'].values
    brake_mask = df['brake'].values

    # Speed vs Time
    plt.figure(figsize=(10,3))
    plt.plot(t, speed, linewidth=1)
    plt.scatter(t[brake_mask], speed[brake_mask], color='red', s=10, label='brake')
    plt.xlabel('Time (s)'); plt.ylabel('Speed (km/h)'); plt.title('Speed vs Time'); plt.grid(True)
    plt.legend()
    fn1 = f"{out_prefix}_speed_time.png"
    plt.tight_layout(); plt.savefig(fn1, dpi=150); plt.close()

    # Elevation vs Distance
    plt.figure(figsize=(10,3))
    plt.plot(dist, ele, linewidth=1)
    plt.xlabel('Distance (m)'); plt.ylabel('Elevation (m)'); plt.title('Elevation vs Distance'); plt.grid(True)
    fn2 = f"{out_prefix}_elev_dist.png"
    plt.tight_layout(); plt.savefig(fn2, dpi=150); plt.close()

    # Speed vs Distance
    plt.figure(figsize=(10,3))
    plt.plot(dist, speed, linewidth=1)
    plt.scatter(dist[brake_mask], speed[brake_mask], color='red', s=10, label='brake')
    plt.xlabel('Distance (m)'); plt.ylabel('Speed (km/h)'); plt.title('Speed vs Distance'); plt.grid(True)
    plt.legend()
    fn3 = f"{out_prefix}_speed_dist.png"
    plt.tight_layout(); plt.savefig(fn3, dpi=150); plt.close()

    return fn1, fn2, fn3

# ----------------------------
# Flask endpoints
# ----------------------------
@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/simulate", methods=['POST'])
def simulate_route():
    if 'gpxfile' not in request.files:
        flash("No file part")
        return redirect(url_for('index'))
    file = request.files['gpxfile']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        uid = uuid.uuid4().hex[:8]
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{fname}")
        file.save(save_path)
        try:
            pts = load_gpx_trackpoints(save_path)
            if len(pts) < 2:
                flash("GPX file doesn't contain enough track points.")
                return redirect(url_for('index'))
            df = simulate_motor_4t(pts, seed=random.randint(0,9999))
            # prepare outputs
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{uid}_{ts}"
            csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base}.csv")
            df.to_csv(csv_path, index=False)
            # make plots
            p1, p2, p3 = make_plots(df, os.path.join(app.config['OUTPUT_FOLDER'], base))
            # render result page with links
            return render_template("result.html",
                                   csv_file=url_for('download_file', filename=os.path.basename(csv_path)),
                                   img1=url_for('download_file', filename=os.path.basename(p1)),
                                   img2=url_for('download_file', filename=os.path.basename(p2)),
                                   img3=url_for('download_file', filename=os.path.basename(p3)),
                                   rows=len(df),
                                   csv_name=os.path.basename(csv_path))
        except Exception as e:
            flash(f"Error processing GPX: {e}")
            return redirect(url_for('index'))
    else:
        flash("File not allowed. Upload a .gpx file.")
        return redirect(url_for('index'))

@app.route("/outputs/<path:filename>", methods=['GET'])
def download_file(filename):
    # serve from OUTPUT_FOLDER, but allow UPLOAD_FOLDER files too if referencing path
    # try outputs then uploads
    outp = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    upl = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(outp):
        return send_file(outp, as_attachment=True)
    elif os.path.exists(upl):
        return send_file(upl, as_attachment=True)
    else:
        return "File not found", 404

# ----------------------------
# TEMPLATES (when not using separate files)
# ----------------------------
# In case you prefer creating files rather than template folder, see README below.
# ----------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5001)
