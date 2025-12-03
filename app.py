import os
import io
import uuid
import math
import random
from datetime import datetime

from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import gpxpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# Config
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXT = {"gpx", "GPX"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Constants from C code
GRAVITY = 9.80655
A_STANDARD = 0.5
K_CONSTANT = 0.02
T_STANDARD = 80.0

def rear_tire_force(s_real, h, v_start, v_end, time_interval):
    if time_interval == 0 or s_real == 0:
        return s_real
    result = ((((v_end - v_start) / time_interval) + (GRAVITY * h / s_real)) / A_STANDARD * s_real)
    return max(0, result)

def front_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase):
    if time_interval == 0 or wheelbase == 0 or s_real == 0:
        return 0.0
    mass_distribution = (0.4 * GRAVITY + (v_start - v_end) / time_interval * 0.55 / wheelbase) / GRAVITY
    normal_mass_distribution = (0.4 * GRAVITY + A_STANDARD * 0.55 / wheelbase) / GRAVITY
    result = mass_distribution * ((((v_start - v_end)) - (GRAVITY * h / s_real)) / (normal_mass_distribution * A_STANDARD) * s_real)
    return max(0, result)

def rear_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase):
    if time_interval == 0 or wheelbase == 0 or s_real == 0:
        return 0.0
    mass_distribution = (0.6 * GRAVITY - (v_start - v_end) / time_interval * 0.55 / wheelbase) / GRAVITY
    normal_mass_distribution = (0.6 * GRAVITY - A_STANDARD * 0.55 / wheelbase) / GRAVITY
    result = mass_distribution * ((((v_start - v_end)) - (GRAVITY * h / s_real)) / (normal_mass_distribution * A_STANDARD) * s_real)
    return max(0, result)

def count_s_oil(s_real, temp_machine):
    return s_real * math.exp(K_CONSTANT * (temp_machine - T_STANDARD))

def calculate_performance_metrics(df, mass, wheelbase, temp_profile):
    perf = {
        's_rear_tire': 0.0, 's_front_tire': 0.0, 's_rear_brake_pad': 0.0,
        's_front_brake_pad': 0.0, 's_chain_or_cvt': 0.0, 's_engine_oil': 0.0,
        's_engine': 0.0, 's_air_filter': 0.0, 'total_distance_km': 0.0,
        'average_speed': 0.0, 'max_speed': 0.0,
    }
    df['speed_m_s'] = df['speed_kmh'] / 3.6
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        s_real = max(0.01, curr_row['dist_m'] - prev_row['dist_m'])
        h = curr_row['ele_m'] - prev_row['ele_m']
        v_start = prev_row['speed_m_s']
        v_end = curr_row['speed_m_s']
        time_interval = max(1, curr_row['time_s'] - prev_row['time_s'])
        is_braking = curr_row['brake']
        
        # Use actual temperature from temp_profile
        current_temp = temp_profile[i] if i < len(temp_profile) else temp_profile[-1]
        
        if is_braking:
            delta_rear_brake = rear_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase)
            delta_front_brake = front_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase)
            perf['s_rear_tire'] += delta_rear_brake
            perf['s_front_tire'] += delta_front_brake
            perf['s_rear_brake_pad'] += delta_rear_brake
            perf['s_front_brake_pad'] += delta_front_brake
            perf['s_chain_or_cvt'] += delta_rear_brake
        else:
            delta_rear_tire = s_real
            if h > 0 and v_end >= v_start:
                delta_rear_tire = rear_tire_force(s_real, h, v_start, v_end, time_interval)
            elif h <= 0 and v_end > v_start:
                delta_rear_tire = rear_tire_force(s_real, h, v_start, v_end, time_interval)
            perf['s_rear_tire'] += delta_rear_tire
            perf['s_front_tire'] += s_real
            perf['s_chain_or_cvt'] += delta_rear_tire
        
        perf['s_engine_oil'] += count_s_oil(s_real, current_temp)
        perf['s_engine'] += s_real
        perf['s_air_filter'] += s_real
        if curr_row['speed_kmh'] > perf['max_speed']:
            perf['max_speed'] = curr_row['speed_kmh']
    
    perf['total_distance_km'] = perf['s_engine'] / 1000.0
    perf['average_speed'] = df['speed_kmh'].mean()
    
    return {
        'total_distance_km': round(perf['total_distance_km'], 2),
        'average_speed_kmh': round(perf['average_speed'], 2),
        'max_speed_kmh': round(perf['max_speed'], 2),
        's_rear_tire_km': round(perf['s_rear_tire'] / 1000.0, 2),
        's_front_tire_km': round(perf['s_front_tire'] / 1000.0, 2),
        's_rear_brake_pad_km': round(perf['s_rear_brake_pad'] / 1000.0, 2),
        's_front_brake_pad_km': round(perf['s_front_brake_pad'] / 1000.0, 2),
        's_chain_or_cvt_km': round(perf['s_chain_or_cvt'] / 1000.0, 2),
        's_engine_oil_km': round(perf['s_engine_oil'] / 1000.0, 2),
        's_engine_km': round(perf['s_engine'] / 1000.0, 2),
        's_air_filter_km': round(perf['s_air_filter'] / 1000.0, 2),
    }

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

def load_gpx_trackpoints(path):
    with open(path, 'r', encoding='utf-8') as f:
        gpx = gpxpy.parse(f)
    pts = []
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

def calculate_total_distance(track_points):
    total = 0.0
    for i in range(len(track_points) - 1):
        lat1, lon1, _ = track_points[i]
        lat2, lon2, _ = track_points[i+1]
        total += haversine_m(lat1, lon1, lat2, lon2)
    return total

def place_events_along_route(total_dist_m, traffic_density, target_time_minutes, seed=0):
    random.seed(seed)
    events = []
    
    if target_time_minutes and target_time_minutes > 0:
        needed_avg_kmh = (total_dist_m / 1000.0) / (target_time_minutes / 60.0)
        if needed_avg_kmh > 50:
            traffic_density = 'low'
        elif needed_avg_kmh > 35:
            traffic_density = 'medium'
    
    if traffic_density == 'low':
        light_freq = (12000, 20000); light_prob = 0.05
        bump_freq = (10000, 18000); bump_prob = 0.06
        macet_count = 0
    elif traffic_density == 'medium':
        light_freq = (4000, 8000); light_prob = 0.15
        bump_freq = (3000, 6000); bump_prob = 0.2
        macet_count = max(0, int(total_dist_m // 25000))
    else:
        light_freq = (2000, 5000); light_prob = 0.25
        bump_freq = (1500, 4000); bump_prob = 0.3
        macet_count = max(1, int(total_dist_m // 12000))
    
    pos = 0
    while pos < total_dist_m:
        pos += random.uniform(*light_freq)
        if pos < total_dist_m and random.random() < light_prob:
            events.append((pos, "lampu_merah"))
    
    pos = 0
    while pos < total_dist_m:
        pos += random.uniform(*bump_freq)
        if pos < total_dist_m and random.random() < bump_prob:
            events.append((pos, "polisi_tidur"))
    
    for _ in range(macet_count):
        events.append((random.uniform(500, max(1000, total_dist_m-500)), "macet"))
    
    return sorted(events, key=lambda x: x[0])

def simulate_motor_4t(track_points, params, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    max_speed_kmh = params.get('max_speed', 100.0)
    min_speed_kmh = params.get('min_speed', 5.0)
    base_cruise_kmh = params.get('avg_speed', 55.0)
    traffic_density = params.get('traffic_density', 'medium')
    target_time_minutes = params.get('target_time_minutes', None)
    
    n = len(track_points)
    if n < 2:
        raise ValueError("GPX must have at least 2 points")
    
    total_dist_m = calculate_total_distance(track_points)
    total_dist_km = total_dist_m / 1000.0
    
    # IMPROVEMENT: Faster speeds for low traffic
    if traffic_density == 'low':
        base_cruise_kmh *= 1.4  # 40% faster on empty roads
        max_speed_kmh = min(max_speed_kmh * 1.2, 150)  # Can go faster
    
    if target_time_minutes and target_time_minutes > 0:
        required_avg_kmh = (total_dist_km) / (target_time_minutes / 60.0)
        base_cruise_kmh = min(max_speed_kmh * 0.95, required_avg_kmh * 1.8)
        base_cruise_kmh = max(min_speed_kmh * 3, base_cruise_kmh)
        print(f"TARGET: {total_dist_km:.1f}km in {target_time_minutes}min needs {required_avg_kmh:.1f} km/h avg, cruise={base_cruise_kmh:.1f}")
    
    max_acc_m_s2 = 4.5  # Slightly faster acceleration
    max_dec_m_s2 = 8.0
    brake_threshold = 1.5
    
    segs = []
    for i in range(n-1):
        lat1, lon1, e1 = track_points[i]
        lat2, lon2, e2 = track_points[i+1]
        d = haversine_m(lat1, lon1, lat2, lon2)
        slope = (e2 - e1) / d if d > 0 else 0.0
        segs.append({
            'i': i, 'lat1': lat1, 'lon1': lon1, 'ele1': e1,
            'lat2': lat2, 'lon2': lon2, 'ele2': e2, 'dist': d, 'slope': slope
        })
    
    events = place_events_along_route(total_dist_m, traffic_density, target_time_minutes, seed=seed)
    
    cum = [0.0]
    for s in segs:
        cum.append(cum[-1] + s['dist'])
    
    seg_event = [None] * len(segs)
    for ev_pos, ev_name in events:
        for j in range(len(segs)):
            if cum[j] <= ev_pos <= cum[j+1]:
                if seg_event[j] is None:
                    seg_event[j] = ev_name
                else:
                    seg_event[j] = seg_event[j] + "|" + ev_name
                break
    
    records = []
    time_s = 0.0
    dist_accum = 0.0
    speed = min(base_cruise_kmh, max_speed_kmh) / 3.6
    
    # Human-like speed variations - reduced for smoother but still realistic
    speed_momentum = 0.0
    hesitation_counter = 0
    
    for idx, s in enumerate(segs):
        slope = s['slope']
        if slope > 0.03:
            slope_penalty = 0.08 + min(0.25, slope * 3.5)  # Less penalty uphill
        elif slope < -0.04:
            slope_penalty = -0.06  # Slight downhill boost
        else:
            slope_penalty = 0.0
        
        angle_rad = 0.0
        i = s['i']
        if 0 < i < len(track_points) - 1:
            a = (track_points[i-1][0], track_points[i-1][1])
            b = (track_points[i][0], track_points[i][1])
            c = (track_points[i+1][0], track_points[i+1][1])
            
            def vec(p, q):
                lat1, lon1 = p; lat2, lon2 = q
                x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1+lat2)/2)) * 6371000
                y = math.radians(lat2 - lat1) * 6371000
                return (x, y)
            
            v1 = vec(b, a); v2 = vec(b, c)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            m1 = math.hypot(v1[0], v1[1]); m2 = math.hypot(v2[0], v2[1])
            if m1*m2 > 0:
                angle_rad = math.acos(max(-1.0, min(1.0, dot/(m1*m2))))
        
        curvature_penalty = (angle_rad / math.radians(90)) * 0.4  # Less penalty for curves
        
        target_kmh = base_cruise_kmh * (1.0 - 0.25 * curvature_penalty - slope_penalty)
        target_kmh = max(min_speed_kmh, min(max_speed_kmh, target_kmh))
        
        event = seg_event[idx]
        if event is not None:
            ev_list = event.split("|")
            if "lampu_merah" in ev_list:
                target_kmh = random.uniform(0.0, 5.0)
            elif "polisi_tidur" in ev_list:
                target_kmh = min(target_kmh, random.uniform(15.0, 25.0))
            elif "macet" in ev_list:
                target_kmh = random.uniform(15.0, 35.0)
        
        if curvature_penalty < 0.05 and abs(slope) < 0.02 and random.random() < 0.08:
            target_kmh = min(max_speed_kmh, target_kmh * random.uniform(1.15, 1.35))
        
        target = target_kmh / 3.6
        segdist = s['dist']
        u = speed
        
        if target > u:
            a = max_acc_m_s2 * random.uniform(0.85, 1.0)
            v_end = min(math.sqrt(max(0.0, u*u + 2*a*segdist)), target)
        else:
            a = max_dec_m_s2 * random.uniform(0.75, 1.0)
            v_end = math.sqrt(max(0.0, u*u - 2*a*segdist))
            if event and ("lampu_merah" in event):
                v_end = min(v_end, target)
        
        avg_v = max(0.5, (u + v_end) / 2.0)
        dt = segdist / avg_v if avg_v > 0 else 1.0
        nsec = max(1, int(round(dt)))
        
        for sidx in range(nsec):
            frac = (sidx + 1) / nsec
            lat = s['lat1'] + (s['lat2'] - s['lat1']) * frac
            lon = s['lon1'] + (s['lon2'] - s['lon1']) * frac
            ele = s['ele1'] + (s['ele2'] - s['ele1']) * frac
            
            # Base speed interpolation
            sp_base = u + (v_end - u) * frac
            
            # REDUCED variations for more consistent speed
            # 1. Momentum (smoother)
            speed_momentum = speed_momentum * 0.90 + random.normalvariate(0, 0.25) * 0.10
            
            # 2. Micro-adjustments (smaller)
            micro_adjust = random.uniform(-0.5, 0.5)
            
            # 3. Occasional hesitations (less frequent)
            hesitation = 0.0
            if random.random() < 0.05:  # 5% chance (was 8%)
                hesitation_counter = random.randint(1, 2)
            if hesitation_counter > 0:
                hesitation = random.uniform(-1.0, -0.3)
                hesitation_counter -= 1
            
            # 4. Speed noise (reduced)
            speed_noise = random.normalvariate(0, 0.2 + sp_base * 0.01)
            
            # Combine all variations
            sp = sp_base + speed_momentum + micro_adjust + hesitation + speed_noise
            
            # Clamp
            sp = max(0.0, min(sp, v_end * 1.12 if v_end > u else v_end * 0.88))
            
            records.append({
                'time_s': time_s, 'lat': lat, 'lon': lon, 'ele_m': ele,
                'speed_m_s': sp, 'dist_m': dist_accum + segdist * frac, 'event': event
            })
            time_s += 1.0
        
        speed = v_end
        dist_accum += segdist
    
    df = pd.DataFrame(records)
    df['speed_kmh'] = df['speed_m_s'] * 3.6
    df['time_s'] = df['time_s'].astype(int)
    df = df[['time_s','lat','lon','ele_m','speed_kmh','dist_m','event','speed_m_s']]
    df = df.reset_index(drop=True)
    
    df['accel_m_s2'] = 0.0
    for i in range(1, len(df)):
        dv = df.loc[i, 'speed_m_s'] - df.loc[i-1, 'speed_m_s']
        dt = max(1.0, df.loc[i, 'time_s'] - df.loc[i-1, 'time_s'])
        df.loc[i, 'accel_m_s2'] = dv / dt
    
    def detect_brake(row):
        if row['accel_m_s2'] < -brake_threshold:
            return True
        ev = row['event']
        if isinstance(ev, str) and ('lampu_merah' in ev or 'sudden_brake' in ev):
            return True
        return False
    
    df['brake'] = df.apply(detect_brake, axis=1)
    df_out = df[['time_s','lat','lon','ele_m','speed_kmh','dist_m','brake','event']].copy()
    df_out['time_s'] = df_out['time_s'].astype(int)
    df_out['brake'] = df_out['brake'].astype(bool)
    df_out = df_out.groupby('time_s', as_index=False).first()
    
    actual_time_min = df_out['time_s'].max() / 60.0
    actual_avg_kmh = (total_dist_km / actual_time_min) * 60.0
    print(f"RESULT: {total_dist_km:.1f}km in {actual_time_min:.1f}min = {actual_avg_kmh:.1f} km/h avg")
    
    return df_out

def calculate_temperature_profile(df, temp_initial, temp_avg, temp_max):
    """
    IMPROVED: Smoother linear temperature progression.
    - Accumulative heating during riding
    - Very slow cooling only when idle
    - Temperature drops significantly only near end
    """
    temp_profile = []
    current_temp = temp_initial
    cumulative_heat = 0.0  # Track accumulated heat
    
    total_time = len(df)
    
    for i in range(len(df)):
        speed_kmh = df.iloc[i]['speed_kmh']
        
        if i > 0:
            is_braking = df.iloc[i]['brake']
            ele_change = df.iloc[i]['ele_m'] - df.iloc[i-1]['ele_m'] if i > 0 else 0
            
            # IMPROVED: More realistic heating model
            # Heat accumulates during riding, cools only when truly idle
            
            if speed_kmh > 70:
                # High speed = significant heating
                heat_gain = 0.25 + (speed_kmh - 70) * 0.01
                target_temp = temp_avg + (temp_max - temp_avg) * 0.8
            elif speed_kmh > 50:
                # Medium-high speed = steady heating
                heat_gain = 0.15
                target_temp = temp_avg + (temp_max - temp_avg) * 0.5
            elif speed_kmh > 30:
                # Medium speed = maintain/slight heating
                heat_gain = 0.08
                target_temp = temp_avg + (temp_max - temp_avg) * 0.2
            elif speed_kmh > 15:
                # Slow speed = minimal heating
                heat_gain = 0.03
                target_temp = temp_avg
            elif speed_kmh > 5:
                # Very slow = start cooling (but slow)
                heat_gain = -0.02
                target_temp = temp_avg - (temp_avg - temp_initial) * 0.2
            else:
                # Idle/stopped = cooling (but only 1-5 degrees)
                # Check if near end of trip
                time_remaining = total_time - i
                if time_remaining < 60:  # Last minute
                    heat_gain = -0.12  # Faster cooling near end
                elif time_remaining < 180:  # Last 3 minutes
                    heat_gain = -0.06
                else:
                    heat_gain = -0.03  # Very slow cooling
                target_temp = current_temp - random.uniform(1, 5)
            
            # Uphill increases heat
            if ele_change > 2.0:
                heat_gain += 0.08 * (ele_change / 10.0)
                target_temp += 3
            
            # Braking reduces heat slightly (but not much)
            if is_braking and speed_kmh > 20:
                heat_gain -= 0.02
            
            # Accumulate heat
            cumulative_heat += heat_gain
            
            # Temperature changes based on accumulated heat
            # Linear progression towards target
            temp_diff = target_temp - current_temp
            current_temp += temp_diff * 0.05 + heat_gain
            
            # Very small random fluctuation (±0.3°C)
            current_temp += random.normalvariate(0, 0.3)
            
            # Enforce limits
            current_temp = max(temp_initial - 3, min(temp_max, current_temp))
        
        temp_profile.append(current_temp)
    
    return temp_profile

def make_plots(df, perf_metrics, temp_profile, mass, wheelbase, temp_params, out_prefix):
    t_sec = df['time_s'].values
    speed = df['speed_kmh'].values
    dist = df['dist_m'].values / 1000.0
    ele = df['ele_m'].values
    brake_mask = df['brake'].values
    
    # Calculate elevation gain/loss
    ele_gain = []
    ele_loss = []
    for i in range(1, len(ele)):
        delta = ele[i] - ele[i-1]
        if delta > 0:
            ele_gain.append(delta)
            ele_loss.append(0)
        else:
            ele_gain.append(0)
            ele_loss.append(abs(delta))
    
    cumulative_gain = np.cumsum([0] + ele_gain)
    cumulative_loss = np.cumsum([0] + ele_loss)
    
    # Plot 1: Speed vs Time
    plt.figure(figsize=(14,5))
    plt.plot(t_sec, speed, linewidth=1.2, color='#2b8aef', alpha=0.7)
    plt.scatter(t_sec[brake_mask], speed[brake_mask], color='red', s=10, label='Brake', alpha=0.5, zorder=5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    plt.title('Speed Profile Over Time - Natural Variation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    fn1 = f"{out_prefix}_speed_time.png"
    plt.tight_layout(); plt.savefig(fn1, dpi=150); plt.close()
    
    # Plot 2: Elevation
    plt.figure(figsize=(14,5))
    plt.fill_between(dist, ele, alpha=0.3, color='#27ae60')
    plt.plot(dist, ele, linewidth=1.5, color='#27ae60')
    plt.xlabel('Distance (km)', fontsize=12, fontweight='bold')
    plt.ylabel('Elevation (m)', fontsize=12, fontweight='bold')
    plt.title('Elevation Profile', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    fn2 = f"{out_prefix}_elev_dist.png"
    plt.tight_layout(); plt.savefig(fn2, dpi=150); plt.close()
    
    # Plot 3: Speed vs Distance
    plt.figure(figsize=(14,5))
    plt.plot(dist, speed, linewidth=1.2, color='#2b8aef', alpha=0.7)
    plt.scatter(dist[brake_mask], speed[brake_mask], color='red', s=10, label='Brake', alpha=0.5, zorder=5)
    plt.xlabel('Distance (km)', fontsize=12, fontweight='bold')
    plt.ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    plt.title('Speed Profile Over Distance - Human-like', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    fn3 = f"{out_prefix}_speed_dist.png"
    plt.tight_layout(); plt.savefig(fn3, dpi=150); plt.close()
    
    # Plot 4: Component Comparison
    plt.figure(figsize=(16,7))
    real_dist = perf_metrics['total_distance_km']
    components = ['Rear\nTire', 'Front\nTire', 'Rear\nBrake', 'Front\nBrake', 'Chain/\nCVT', 'Oil\nEquiv', 'Engine', 'Air\nFilter']
    values = [
        perf_metrics['s_rear_tire_km'], perf_metrics['s_front_tire_km'],
        perf_metrics['s_rear_brake_pad_km'], perf_metrics['s_front_brake_pad_km'],
        perf_metrics['s_chain_or_cvt_km'], perf_metrics['s_engine_oil_km'],
        perf_metrics['s_engine_km'], perf_metrics['s_air_filter_km']
    ]
    
    x_pos = np.arange(len(components))
    colors = ['#e74c3c', '#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12', '#2ecc71', '#95a5a6']
    bars = plt.bar(x_pos, values, color=colors, alpha=0.75, edgecolor='black', linewidth=1.5)
    plt.axhline(y=real_dist, color='#000000', linestyle='--', linewidth=3, label=f'Real Distance ({real_dist} km)', zorder=10)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        diff = val - real_dist
        sign = '+' if diff > 0 else ''
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}\n({sign}{diff:.1f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Components', fontsize=13, fontweight='bold')
    plt.ylabel('Distance Equivalent (km)', fontsize=13, fontweight='bold')
    plt.title('Component Wear vs Real Distance', fontsize=15, fontweight='bold')
    plt.xticks(x_pos, components, fontsize=11)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    fn4 = f"{out_prefix}_component_comparison.png"
    plt.tight_layout(); plt.savefig(fn4, dpi=150); plt.close()
    
    # Plot 5: Brake Distribution
    plt.figure(figsize=(14,6))
    brake_data = [perf_metrics['s_front_brake_pad_km'], perf_metrics['s_rear_brake_pad_km']]
    
    plt.subplot(1, 2, 1)
    colors_pie = ['#3498db', '#e74c3c']
    explode = (0.05, 0)
    plt.pie(brake_data, labels=['Front Brake', 'Rear Brake'], autopct='%1.1f%%',
            colors=colors_pie, explode=explode, shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Brake Force Distribution', fontsize=13, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    brake_components = ['Front Brake\nPad', 'Rear Brake\nPad']
    x_pos = np.arange(len(brake_components))
    bars = plt.bar(x_pos, brake_data, color=['#3498db', '#e74c3c'], alpha=0.75, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, brake_data):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.2f} km', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Brake Component', fontsize=12, fontweight='bold')
    plt.ylabel('Wear Equivalent (km)', fontsize=12, fontweight='bold')
    plt.title('Brake Pad Wear Comparison', fontsize=13, fontweight='bold')
    plt.xticks(x_pos, brake_components, fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    fn5 = f"{out_prefix}_brake_analysis.png"
    plt.tight_layout(); plt.savefig(fn5, dpi=150); plt.close()
    
    # Plot 6: Real-time ALL 8 Components
    plt.figure(figsize=(20,11))
    
    cumulative = {
        'actual': [], 'rear_tire': [], 'front_tire': [], 'rear_brake': [],
        'front_brake': [], 'chain_cvt': [], 'oil': [], 'engine': [], 'air_filter': []
    }
    
    cum_vals = {k: 0.0 for k in cumulative.keys()}
    df['speed_m_s'] = df['speed_kmh'] / 3.6
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        s_real = max(0.01, curr_row['dist_m'] - prev_row['dist_m'])
        h = curr_row['ele_m'] - prev_row['ele_m']
        v_start = prev_row['speed_m_s']
        v_end = curr_row['speed_m_s']
        time_interval = max(1, curr_row['time_s'] - prev_row['time_s'])
        current_temp = temp_profile[i]
        
        if curr_row['brake']:
            delta_rear_brake = rear_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase)
            delta_front_brake = front_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase)
            
            cum_vals['rear_tire'] += delta_rear_brake
            cum_vals['front_tire'] += delta_front_brake
            cum_vals['rear_brake'] += delta_rear_brake
            cum_vals['front_brake'] += delta_front_brake
            cum_vals['chain_cvt'] += delta_rear_brake
        else:
            delta_rear = s_real
            if h > 0 and v_end >= v_start:
                delta_rear = rear_tire_force(s_real, h, v_start, v_end, time_interval)
            elif h <= 0 and v_end > v_start:
                delta_rear = rear_tire_force(s_real, h, v_start, v_end, time_interval)
            
            cum_vals['rear_tire'] += delta_rear
            cum_vals['front_tire'] += s_real
            cum_vals['chain_cvt'] += delta_rear
        
        cum_vals['oil'] += count_s_oil(s_real, current_temp)
        cum_vals['engine'] += s_real
        cum_vals['air_filter'] += s_real
        cum_vals['actual'] = curr_row['dist_m'] / 1000.0
        
        for key in cumulative.keys():
            cumulative[key].append(cum_vals[key] / 1000.0 if key != 'actual' else cum_vals['actual'])
    
    time_plot = df['time_s'].values[1:]
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle('Real-time Tracking: ALL 8 COMPONENTS', fontsize=18, fontweight='bold')
    
    components_data = [
        ('rear_tire', 'Rear Tire', '#e74c3c'),
        ('front_tire', 'Front Tire', '#3498db'),
        ('rear_brake', 'Rear Brake Pad', '#e67e22'),
        ('front_brake', 'Front Brake Pad', '#9b59b6'),
        ('chain_cvt', 'Chain/CVT', '#1abc9c'),
        ('oil', 'Engine Oil', '#f39c12'),
        ('engine', 'Engine', '#2ecc71'),
        ('air_filter', 'Air Filter', '#95a5a6')
    ]
    
    for idx, (key, label, color) in enumerate(components_data):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        ax.plot(time_plot, cumulative['actual'], linewidth=2.5, color='black', label='Actual', alpha=0.9, linestyle='--')
        ax.plot(time_plot, cumulative[key], linewidth=2.2, color=color, label=label, alpha=0.85)
        
        final_diff = cumulative[key][-1] - cumulative['actual'][-1]
        ax.text(0.5, 0.95, f'Diff: {final_diff:+.2f} km', transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
        
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Distance (km)', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.4, linestyle='--')
    
    fn6 = f"{out_prefix}_realtime_all_8.png"
    plt.tight_layout()
    plt.savefig(fn6, dpi=150)
    plt.close()
    
    # Plot 7: Temperature & Oil (SMOOTH LINEAR)
    plt.figure(figsize=(16,9))
    
    oil_wear_rate = []
    for temp in temp_profile:
        oil_factor = math.exp(K_CONSTANT * (temp - T_STANDARD))
        oil_wear_rate.append(oil_factor)
    
    plt.subplot(2, 1, 1)
    plt.plot(t_sec, temp_profile, linewidth=2.5, color='#e74c3c', alpha=0.9, label='Actual Temperature')
    plt.axhline(y=temp_params['temp_avg'], color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Target Avg ({temp_params["temp_avg"]}°C)')
    plt.axhline(y=temp_params['temp_initial'], color='#3498db', linestyle='--', linewidth=2, label=f'Initial ({temp_params["temp_initial"]}°C)')
    plt.axhline(y=temp_params['temp_max'], color='#e74c3c', linestyle='--', linewidth=2, label=f'Max Limit ({temp_params["temp_max"]}°C)')
    plt.fill_between(t_sec, temp_params['temp_initial'], temp_profile, alpha=0.25, color='#e74c3c')
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    plt.title('Engine Temperature - Smooth Linear Progression (Accumulative Heating)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(temp_params['temp_initial'] - 10, temp_params['temp_max'] + 5)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_sec, oil_wear_rate, linewidth=2.2, color='#f39c12', alpha=0.85, label='Oil Degradation Rate')
    plt.axhline(y=1.0, color='#2ecc71', linestyle='--', linewidth=2.5, label='Standard (1.0x at 90°C)')
    plt.fill_between(t_sec, 0, oil_wear_rate, alpha=0.25, color='#f39c12')
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('Oil Wear Factor', fontsize=13, fontweight='bold')
    plt.title('Oil Degradation Rate', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    
    avg_temp = np.mean(temp_profile)
    max_temp = np.max(temp_profile)
    min_temp = np.min(temp_profile)
    
    stats_text = f"Temp: Avg={avg_temp:.1f}°C Max={max_temp:.1f}°C Min={min_temp:.1f}°C | Smooth linear with accumulative heating"
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    fn7 = f"{out_prefix}_temperature_smooth.png"
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fn7, dpi=150)
    plt.close()
    
    # Plot 8: Combined Line Chart
    plt.figure(figsize=(18, 10))
    
    time_plot_all = df['time_s'].values[1:]
    
    plt.plot(time_plot_all, cumulative['actual'], linewidth=3.5, color='#000000', 
             label='Actual Distance', alpha=1.0, linestyle='-', zorder=10)
    
    components_line = [
        ('rear_tire', 'Rear Tire', '#e74c3c', '--'),
        ('front_tire', 'Front Tire', '#3498db', '--'),
        ('rear_brake', 'Rear Brake Pad', '#e67e22', '-.'),
        ('front_brake', 'Front Brake Pad', '#9b59b6', '-.'),
        ('chain_cvt', 'Chain/CVT', '#1abc9c', ':'),
        ('oil', 'Engine Oil', '#f39c12', ':'),
        ('engine', 'Engine', '#2ecc71', '-'),
        ('air_filter', 'Air Filter', '#95a5a6', '-')
    ]
    
    for key, label, color, linestyle in components_line:
        plt.plot(time_plot_all, cumulative[key], linewidth=2.2, color=color, 
                label=label, alpha=0.75, linestyle=linestyle)
    
    plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Distance (km)', fontsize=14, fontweight='bold')
    plt.title('ALL COMPONENTS vs ACTUAL - Combined Line Chart', fontsize=16, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='upper left', ncol=2, framealpha=0.95, edgecolor='black', shadow=True)
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    fn8 = f"{out_prefix}_all_combined.png"
    plt.tight_layout()
    plt.savefig(fn8, dpi=150)
    plt.close()
    
    # Plot 9: NEW - Brake vs Elevation Gain
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    brake_times = t_sec[1:][brake_mask[1:]]
    brake_elevations = ele[1:][brake_mask[1:]]
    plt.scatter(brake_elevations, brake_times, c='red', s=30, alpha=0.6, edgecolors='darkred')
    plt.xlabel('Elevation (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Brake Events vs Elevation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dist[1:], cumulative_gain[1:], color='green', linewidth=2, label='Cumulative Gain', alpha=0.8)
    plt.plot(dist[1:], cumulative_loss[1:], color='orange', linewidth=2, label='Cumulative Loss', alpha=0.8)
    brake_dist = dist[1:][brake_mask[1:]]
    brake_gain_at_brake = []
    for bd in brake_dist:
        idx = np.argmin(np.abs(dist[1:] - bd))
        brake_gain_at_brake.append(cumulative_gain[1:][idx])
    plt.scatter(brake_dist, brake_gain_at_brake, c='red', s=40, alpha=0.7, edgecolors='darkred', label='Brakes', zorder=5)
    plt.xlabel('Distance (km)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Elevation Change (m)', fontsize=12, fontweight='bold')
    plt.title('Brake Events vs Elevation Gain/Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fn9 = f"{out_prefix}_brake_vs_elevation.png"
    plt.tight_layout()
    plt.savefig(fn9, dpi=150)
    plt.close()
    
    # Plot 10: NEW - Speed vs Elevation Profile (Overlay)
    plt.figure(figsize=(16, 8))
    
    ax1 = plt.gca()
    color = 'tab:blue'
    ax1.set_xlabel('Distance (km)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Speed (km/h)', color=color, fontsize=13, fontweight='bold')
    ax1.plot(dist, speed, color=color, linewidth=2, alpha=0.8, label='Speed')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Elevation (m)', color=color, fontsize=13, fontweight='bold')
    ax2.fill_between(dist, ele, alpha=0.3, color=color)
    ax2.plot(dist, ele, color=color, linewidth=1.5, alpha=0.8, label='Elevation')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Speed vs Elevation Profile - Overlay', fontsize=15, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    fn10 = f"{out_prefix}_speed_vs_elevation.png"
    plt.tight_layout()
    plt.savefig(fn10, dpi=150)
    plt.close()
    
    # Plot 11: NEW - Component Usage Based on Elevation
    plt.figure(figsize=(18, 10))
    
    # Divide route into elevation zones
    ele_bins = np.linspace(ele.min(), ele.max(), 6)
    ele_labels = [f'{ele_bins[i]:.0f}-{ele_bins[i+1]:.0f}m' for i in range(len(ele_bins)-1)]
    
    # Calculate component wear in each zone
    zone_wear = {comp: [0]*5 for comp in ['rear_tire', 'front_tire', 'rear_brake', 'front_brake', 'chain_cvt', 'oil', 'engine', 'air_filter']}
    
    df['speed_m_s'] = df['speed_kmh'] / 3.6
    
    for i in range(1, len(df)):
        curr_ele = df.iloc[i]['ele_m']
        zone_idx = min(4, max(0, np.digitize(curr_ele, ele_bins) - 1))
        
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        s_real = max(0.01, curr_row['dist_m'] - prev_row['dist_m'])
        h = curr_row['ele_m'] - prev_row['ele_m']
        v_start = prev_row['speed_m_s']
        v_end = curr_row['speed_m_s']
        time_interval = max(1, curr_row['time_s'] - prev_row['time_s'])
        current_temp = temp_profile[i]
        
        if curr_row['brake']:
            delta_rear_brake = rear_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase)
            delta_front_brake = front_brake_work(s_real, h, v_start, v_end, time_interval, mass, wheelbase)
            zone_wear['rear_tire'][zone_idx] += delta_rear_brake / 1000.0
            zone_wear['front_tire'][zone_idx] += delta_front_brake / 1000.0
            zone_wear['rear_brake'][zone_idx] += delta_rear_brake / 1000.0
            zone_wear['front_brake'][zone_idx] += delta_front_brake / 1000.0
            zone_wear['chain_cvt'][zone_idx] += delta_rear_brake / 1000.0
        else:
            delta_rear = s_real
            if h > 0 and v_end >= v_start:
                delta_rear = rear_tire_force(s_real, h, v_start, v_end, time_interval)
            elif h <= 0 and v_end > v_start:
                delta_rear = rear_tire_force(s_real, h, v_start, v_end, time_interval)
            zone_wear['rear_tire'][zone_idx] += delta_rear / 1000.0
            zone_wear['front_tire'][zone_idx] += s_real / 1000.0
            zone_wear['chain_cvt'][zone_idx] += delta_rear / 1000.0
        
        zone_wear['oil'][zone_idx] += count_s_oil(s_real, current_temp) / 1000.0
        zone_wear['engine'][zone_idx] += s_real / 1000.0
        zone_wear['air_filter'][zone_idx] += s_real / 1000.0
    
    # Plot stacked bar chart
    x = np.arange(len(ele_labels))
    width = 0.15
    
    colors_comp = ['#e74c3c', '#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12', '#2ecc71', '#95a5a6']
    comp_names = ['Rear Tire', 'Front Tire', 'Rear Brake', 'Front Brake', 'Chain/CVT', 'Oil', 'Engine', 'Air Filter']
    
    for idx, (comp, name) in enumerate(zip(['rear_tire', 'front_tire', 'rear_brake', 'front_brake', 'chain_cvt', 'oil', 'engine', 'air_filter'], comp_names)):
        plt.bar(x + idx*width, zone_wear[comp], width, label=name, color=colors_comp[idx], alpha=0.8)
    
    plt.xlabel('Elevation Zone', fontsize=13, fontweight='bold')
    plt.ylabel('Component Wear (km equivalent)', fontsize=13, fontweight='bold')
    plt.title('Component Usage Based on Elevation Zones', fontsize=15, fontweight='bold')
    plt.xticks(x + width*3.5, ele_labels, fontsize=10)
    plt.legend(ncol=2, fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    fn11 = f"{out_prefix}_component_by_elevation.png"
    plt.tight_layout()
    plt.savefig(fn11, dpi=150)
    plt.close()
    
    # Plot 12: NEW - Speed Distribution by Elevation Zones
    plt.figure(figsize=(16, 8))
    
    zone_speeds = {i: [] for i in range(5)}
    for i in range(len(df)):
        curr_ele = df.iloc[i]['ele_m']
        zone_idx = min(4, max(0, np.digitize(curr_ele, ele_bins) - 1))
        zone_speeds[zone_idx].append(df.iloc[i]['speed_kmh'])
    
    plt.subplot(1, 2, 1)
    bp = plt.boxplot([zone_speeds[i] for i in range(5)], labels=ele_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db']*5):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xlabel('Elevation Zone', fontsize=12, fontweight='bold')
    plt.ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    plt.title('Speed Distribution by Elevation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    plt.subplot(1, 2, 2)
    avg_speeds = [np.mean(zone_speeds[i]) for i in range(5)]
    plt.bar(ele_labels, avg_speeds, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xlabel('Elevation Zone', fontsize=12, fontweight='bold')
    plt.ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
    plt.title('Average Speed by Elevation Zone', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    for i, v in enumerate(avg_speeds):
        plt.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
    
    fn12 = f"{out_prefix}_speed_by_elevation.png"
    plt.tight_layout()
    plt.savefig(fn12, dpi=150)
    plt.close()
    
    return fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11, fn12

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/simulate", methods=['POST'])
def simulate_route():
    if 'gpxfile' not in request.files:
        flash("No file uploaded")
        return redirect(url_for('index'))
    
    file = request.files['gpxfile']
    if file.filename == '':
        flash("No file selected")
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            max_speed = float(request.form.get('max_speed', 100))
            min_speed = float(request.form.get('min_speed', 5))
            avg_speed = float(request.form.get('avg_speed', 55))
            traffic_density = request.form.get('traffic_density', 'medium')
            target_time_minutes = request.form.get('target_time_minutes', '')
            mass = float(request.form.get('mass', 150))
            wheelbase = float(request.form.get('wheelbase', 1.3))
            
            temp_initial = float(request.form.get('temp_initial', 70))
            temp_avg = float(request.form.get('temp_avg', 95))
            temp_max = float(request.form.get('temp_max', 120))
            
            if not (temp_initial < temp_avg < temp_max):
                flash("Temperature parameters must satisfy: Initial < Average < Max")
                return redirect(url_for('index'))
            
            if target_time_minutes and target_time_minutes.strip():
                try:
                    target_time_minutes = float(target_time_minutes)
                    if target_time_minutes <= 0:
                        flash("Target time must be greater than 0")
                        return redirect(url_for('index'))
                except ValueError:
                    flash("Invalid target time value")
                    return redirect(url_for('index'))
            else:
                target_time_minutes = None
            
            if max_speed <= min_speed:
                flash("Max speed must be greater than min speed")
                return redirect(url_for('index'))
            if not (min_speed <= avg_speed <= max_speed):
                flash("Average speed must be between min and max speed")
                return redirect(url_for('index'))
            
            fname = secure_filename(file.filename)
            uid = uuid.uuid4().hex[:8]
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{fname}")
            file.save(save_path)
            
            pts = load_gpx_trackpoints(save_path)
            if len(pts) < 2:
                flash("GPX file doesn't contain enough track points")
                return redirect(url_for('index'))
            
            params = {
                'max_speed': max_speed, 'min_speed': min_speed, 'avg_speed': avg_speed,
                'traffic_density': traffic_density, 'target_time_minutes': target_time_minutes
            }
            
            print(f"\n{'='*70}")
            print(f"SIMULATION STARTING...")
            print(f"Max Speed: {max_speed} km/h | Traffic: {traffic_density}")
            print(f"Temperature: {temp_initial}°C → {temp_avg}°C → max {temp_max}°C")
            print(f"Target Time: {target_time_minutes if target_time_minutes else 'Not set'} min")
            print(f"{'='*70}\n")
            
            df = simulate_motor_4t(pts, params, seed=random.randint(0, 9999))
            temp_profile = calculate_temperature_profile(df, temp_initial, temp_avg, temp_max)
            perf_metrics = calculate_performance_metrics(df, mass, wheelbase, temp_profile)
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{uid}_{ts}"
            csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base}.csv")
            df.to_csv(csv_path, index=False)
            
            temp_params = {'temp_initial': temp_initial, 'temp_avg': temp_avg, 'temp_max': temp_max}
            
            plots = make_plots(df, perf_metrics, temp_profile, mass, wheelbase, temp_params,
                              os.path.join(app.config['OUTPUT_FOLDER'], base))
            
            total_time_min = df['time_s'].max() / 60.0
            total_time_hr = total_time_min / 60.0
            
            time_warning = None
            if target_time_minutes:
                diff_pct = abs(total_time_min - target_time_minutes) / target_time_minutes * 100
                if diff_pct > 20:
                    time_warning = f"⚠️ Actual time ({total_time_min:.1f} min) differs {diff_pct:.0f}% from target"
            
            avg_temp = np.mean(temp_profile)
            max_temp_actual = np.max(temp_profile)
            min_temp_actual = np.min(temp_profile)
            
            print(f"\n{'='*70}")
            print(f"SIMULATION COMPLETE! 12 PLOTS GENERATED")
            print(f"Distance: {perf_metrics['total_distance_km']} km")
            print(f"Time: {total_time_min:.1f} min ({total_time_hr:.2f} hrs)")
            print(f"Avg Speed: {perf_metrics['average_speed_kmh']} km/h")
            print(f"Temperature: Avg={avg_temp:.1f}°C, Max={max_temp_actual:.1f}°C")
            print(f"{'='*70}\n")
            
            return render_template("result.html",
                                   csv_file=url_for('download_file', filename=os.path.basename(csv_path)),
                                   img1=url_for('download_file', filename=os.path.basename(plots[0])),
                                   img2=url_for('download_file', filename=os.path.basename(plots[1])),
                                   img3=url_for('download_file', filename=os.path.basename(plots[2])),
                                   img4=url_for('download_file', filename=os.path.basename(plots[3])),
                                   img5=url_for('download_file', filename=os.path.basename(plots[4])),
                                   img6=url_for('download_file', filename=os.path.basename(plots[5])),
                                   img7=url_for('download_file', filename=os.path.basename(plots[6])),
                                   img8=url_for('download_file', filename=os.path.basename(plots[7])),
                                   img9=url_for('download_file', filename=os.path.basename(plots[8])),
                                   img10=url_for('download_file', filename=os.path.basename(plots[9])),
                                   img11=url_for('download_file', filename=os.path.basename(plots[10])),
                                   img12=url_for('download_file', filename=os.path.basename(plots[11])),
                                   rows=len(df), csv_name=os.path.basename(csv_path),
                                   perf=perf_metrics, total_time_min=round(total_time_min, 2),
                                   total_time_hr=round(total_time_hr, 2), time_warning=time_warning,
                                   temp_stats={'avg': round(avg_temp, 1), 'max': round(max_temp_actual, 1), 'min': round(min_temp_actual, 1)},
                                   params={
                                       'max_speed': max_speed, 'min_speed': min_speed, 'avg_speed': avg_speed,
                                       'traffic_density': traffic_density, 'target_time_minutes': target_time_minutes if target_time_minutes else 'Not set',
                                       'mass': mass, 'wheelbase': wheelbase, 'temp_initial': temp_initial, 'temp_avg': temp_avg, 'temp_max': temp_max
                                   })
        except Exception as e:
            flash(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return redirect(url_for('index'))
    else:
        flash("Invalid file type")
        return redirect(url_for('index'))

@app.route("/outputs/<path:filename>", methods=['GET'])
def download_file(filename):
    outp = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(outp):
        return send_file(outp, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=5001)