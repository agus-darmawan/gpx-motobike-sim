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
GRAVITY = 9.81
A_STANDARD = 1.0
K_CONSTANT = 0.05
T_STANDARD = 90.0

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

def calculate_performance_metrics(df, mass, wheelbase, temp_machine):
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
        
        perf['s_engine_oil'] += count_s_oil(s_real, temp_machine)
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
        light_freq = (8000, 15000); light_prob = 0.08
        bump_freq = (6000, 12000); bump_prob = 0.1
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
    
    if target_time_minutes and target_time_minutes > 0:
        required_avg_kmh = (total_dist_km) / (target_time_minutes / 60.0)
        base_cruise_kmh = min(max_speed_kmh * 0.95, required_avg_kmh * 1.8)
        base_cruise_kmh = max(min_speed_kmh * 3, base_cruise_kmh)
        print(f"TARGET: {total_dist_km:.1f}km in {target_time_minutes}min needs {required_avg_kmh:.1f} km/h avg, cruise={base_cruise_kmh:.1f}")
    
    max_acc_m_s2 = 4.0
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
    
    for idx, s in enumerate(segs):
        slope = s['slope']
        if slope > 0.03:
            slope_penalty = 0.1 + min(0.3, slope * 4)
        elif slope < -0.04:
            slope_penalty = -0.08
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
        
        curvature_penalty = (angle_rad / math.radians(90)) * 0.5
        
        target_kmh = base_cruise_kmh * (1.0 - 0.3 * curvature_penalty - slope_penalty)
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
        
        if curvature_penalty < 0.05 and abs(slope) < 0.02 and random.random() < 0.05:
            target_kmh = min(max_speed_kmh, target_kmh * random.uniform(1.2, 1.4))
        
        target = target_kmh / 3.6
        segdist = s['dist']
        u = speed
        
        if target > u:
            a = max_acc_m_s2 * random.uniform(0.8, 1.0)
            v_end = min(math.sqrt(max(0.0, u*u + 2*a*segdist)), target)
        else:
            a = max_dec_m_s2 * random.uniform(0.7, 1.0)
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
            sp = max(0.0, (u + (v_end - u) * frac) + random.normalvariate(0, 0.15))
            
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

def make_plots(df, perf_metrics, temp_machine, mass, wheelbase, out_prefix):
    t_sec = df['time_s'].values
    speed = df['speed_kmh'].values
    dist = df['dist_m'].values / 1000.0
    ele = df['ele_m'].values
    brake_mask = df['brake'].values
    
    # Plot 1: Speed vs Time (PER SECOND)
    plt.figure(figsize=(14,5))
    plt.plot(t_sec, speed, linewidth=1.2, color='#2b8aef', alpha=0.7)
    plt.scatter(t_sec[brake_mask], speed[brake_mask], color='red', s=10, label='Brake', alpha=0.5, zorder=5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    plt.title('Speed Profile Over Time (Per Second)', fontsize=14, fontweight='bold')
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
    plt.title('Speed Profile Over Distance (Per Second)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    fn3 = f"{out_prefix}_speed_dist.png"
    plt.tight_layout(); plt.savefig(fn3, dpi=150); plt.close()
    
    # Plot 4: Component Comparison - ALL 8
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
    
    plt.xlabel('Components (ALL 8)', fontsize=13, fontweight='bold')
    plt.ylabel('Distance Equivalent (km)', fontsize=13, fontweight='bold')
    plt.title('Component Wear vs Real Distance - ALL 8 COMPONENTS', fontsize=15, fontweight='bold')
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
    
    # Plot 6: ALL 8 COMPONENTS Real-time Tracking (PER SECOND)
    plt.figure(figsize=(20,11))
    
    # Calculate cumulative for ALL 8 components
    cumulative = {
        'actual': [],
        'rear_tire': [],
        'front_tire': [],
        'rear_brake': [],
        'front_brake': [],
        'chain_cvt': [],
        'oil': [],
        'engine': [],
        'air_filter': []
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
        
        cum_vals['oil'] += count_s_oil(s_real, temp_machine)
        cum_vals['engine'] += s_real
        cum_vals['air_filter'] += s_real
        cum_vals['actual'] = curr_row['dist_m'] / 1000.0
        
        for key in cumulative.keys():
            cumulative[key].append(cum_vals[key] / 1000.0 if key != 'actual' else cum_vals['actual'])
    
    time_plot = df['time_s'].values[1:]
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle('Real-time Tracking: ALL 8 COMPONENTS (Per Second)', fontsize=18, fontweight='bold')
    
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
    
    fn6 = f"{out_prefix}_realtime_all_8_components.png"
    plt.tight_layout()
    plt.savefig(fn6, dpi=150)
    plt.close()
    
    # Plot 7: Temperature & Oil Per Second
    plt.figure(figsize=(16,9))
    
    temp_profile = []
    oil_wear_rate = []
    
    for i in range(len(df)):
        curr_speed = df.iloc[i]['speed_kmh']
        if i > 0:
            if curr_speed > 80:
                temp_var = temp_machine + random.uniform(0, 12)
            elif curr_speed > 60:
                temp_var = temp_machine + random.uniform(-5, 8)
            elif curr_speed > 30:
                temp_var = temp_machine + random.uniform(-10, 3)
            else:
                temp_var = temp_machine + random.uniform(-15, -3)
            
            if len(temp_profile) > 0:
                temp_var = temp_profile[-1] * 0.85 + temp_var * 0.15
        else:
            temp_var = temp_machine
        
        temp_profile.append(max(60, min(120, temp_var)))
        oil_factor = math.exp(K_CONSTANT * (temp_profile[-1] - T_STANDARD))
        oil_wear_rate.append(oil_factor)
    
    plt.subplot(2, 1, 1)
    plt.plot(t_sec, temp_profile, linewidth=2.2, color='#e74c3c', alpha=0.85)
    plt.axhline(y=temp_machine, color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Average ({temp_machine}°C)')
    plt.axhline(y=T_STANDARD, color='#3498db', linestyle='--', linewidth=2.5, label=f'Standard ({T_STANDARD}°C)')
    plt.fill_between(t_sec, 60, temp_profile, alpha=0.25, color='#e74c3c')
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
    plt.title('Engine Temperature Profile (Per Second)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    plt.ylim(60, 120)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_sec, oil_wear_rate, linewidth=2.2, color='#f39c12', alpha=0.85)
    plt.axhline(y=1.0, color='#2ecc71', linestyle='--', linewidth=2.5, label='Standard (1.0x)')
    plt.fill_between(t_sec, 0, oil_wear_rate, alpha=0.25, color='#f39c12')
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('Oil Wear Factor', fontsize=13, fontweight='bold')
    plt.title('Oil Degradation Rate (Per Second)', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    avg_temp = np.mean(temp_profile)
    max_temp = np.max(temp_profile)
    min_temp = np.min(temp_profile)
    avg_oil_rate = np.mean(oil_wear_rate)
    max_oil_rate = np.max(oil_wear_rate)
    min_oil_rate = np.min(oil_wear_rate)
    
    stats_text = f"Temperature: Avg={avg_temp:.1f}°C Max={max_temp:.1f}°C Min={min_temp:.1f}°C  |  Oil Rate: Avg={avg_oil_rate:.2f}x Max={max_oil_rate:.2f}x Min={min_oil_rate:.2f}x"
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    fn7 = f"{out_prefix}_temperature_oil_per_second.png"
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fn7, dpi=150)
    plt.close()
    
    return fn1, fn2, fn3, fn4, fn5, fn6, fn7

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
            temp_machine = float(request.form.get('temp_machine', 95))
            
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
            print(f"Max Speed: {max_speed} km/h")
            print(f"Target Time: {target_time_minutes if target_time_minutes else 'Not set'} minutes")
            print(f"{'='*70}\n")
            
            df = simulate_motor_4t(pts, params, seed=random.randint(0, 9999))
            perf_metrics = calculate_performance_metrics(df, mass, wheelbase, temp_machine)
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{uid}_{ts}"
            csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base}.csv")
            df.to_csv(csv_path, index=False)
            
            p1, p2, p3, p4, p5, p6, p7 = make_plots(df, perf_metrics, temp_machine, mass, wheelbase,
                                                     os.path.join(app.config['OUTPUT_FOLDER'], base))
            
            total_time_min = df['time_s'].max() / 60.0
            total_time_hr = total_time_min / 60.0
            
            time_warning = None
            if target_time_minutes:
                diff_pct = abs(total_time_min - target_time_minutes) / target_time_minutes * 100
                if diff_pct > 20:
                    time_warning = f"⚠️ Actual time ({total_time_min:.1f} min) differs {diff_pct:.0f}% from target ({target_time_minutes:.1f} min)"
            
            print(f"\n{'='*70}")
            print(f"SIMULATION COMPLETE!")
            print(f"Distance: {perf_metrics['total_distance_km']} km")
            print(f"Time: {total_time_min:.1f} min ({total_time_hr:.2f} hrs)")
            print(f"Avg Speed: {perf_metrics['average_speed_kmh']} km/h")
            print(f"Max Speed: {perf_metrics['max_speed_kmh']} km/h")
            print(f"{'='*70}\n")
            
            return render_template("result.html",
                                   csv_file=url_for('download_file', filename=os.path.basename(csv_path)),
                                   img1=url_for('download_file', filename=os.path.basename(p1)),
                                   img2=url_for('download_file', filename=os.path.basename(p2)),
                                   img3=url_for('download_file', filename=os.path.basename(p3)),
                                   img4=url_for('download_file', filename=os.path.basename(p4)),
                                   img5=url_for('download_file', filename=os.path.basename(p5)),
                                   img6=url_for('download_file', filename=os.path.basename(p6)),
                                   img7=url_for('download_file', filename=os.path.basename(p7)),
                                   rows=len(df), csv_name=os.path.basename(csv_path),
                                   perf=perf_metrics, total_time_min=round(total_time_min, 2),
                                   total_time_hr=round(total_time_hr, 2), time_warning=time_warning,
                                   params={
                                       'max_speed': max_speed, 'min_speed': min_speed,
                                       'avg_speed': avg_speed, 'traffic_density': traffic_density,
                                       'target_time_minutes': target_time_minutes if target_time_minutes else 'Not set',
                                       'mass': mass, 'wheelbase': wheelbase, 'temp_machine': temp_machine
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
    app.run(debug=True, port=5001)