import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from math import radians
import time

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False
    
import matplotlib
matplotlib.use("QtAgg")

Q = 0.1; U = 3.0; H = 5.0
ay, by = 0.32, 0.78
az, bz = 0.24, 0.67

x_min, x_max, nx = 0.0, 200.0, 160
y_min, y_max, ny = -80.0, 80.0, 160
z_min, z_max, nz = 0.0, 80.0, 80

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

x_pos = np.maximum(X, 1e-3)
sigma_y = ay * (x_pos**by)
sigma_z = az * (x_pos**bz)

prefactor = Q / (U * 2.0 * np.pi * sigma_y * sigma_z)
gauss_y = np.exp(-(Y**2) / (2.0 * sigma_y**2))
gauss_z = np.exp(-((Z - H)**2) / (2.0 * sigma_z**2)) + np.exp(-((Z + H)**2) / (2.0 * sigma_z**2))
k_ppm = 1e9
VOL = (k_ppm * prefactor * gauss_y * gauss_z).astype(np.float32)
BOUNDS = np.array([x_min, x_max, y_min, y_max, z_min, z_max], dtype=np.float32)

def look_at(camera_pos, target, up_vec):
    cam_pos = np.array(camera_pos, dtype=np.float32)
    tgt = np.array(target, dtype=np.float32)
    up = np.array(up_vec, dtype=np.float32)
    fwd = tgt - cam_pos; fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up); right = right / np.linalg.norm(right)
    upn = np.cross(right, fwd)
    return right.astype(np.float32), upn.astype(np.float32), fwd.astype(np.float32)

def make_dirs(img_w, img_h, fov_deg, right, up, fwd):
    fov = radians(float(fov_deg))
    aspect = img_w / img_h
    plane_h = 2.0 * np.tan(fov/2.0)
    plane_w = aspect * plane_h
    ys, xs = np.meshgrid(
        np.linspace(-0.5, 0.5, img_h, endpoint=False) + (0.5/img_h),
        np.linspace(-0.5, 0.5, img_w, endpoint=False) + (0.5/img_w),
        indexing='ij'
    )
    px = xs * plane_w
    py = ys * plane_h
    dirs = (fwd[None,None,:] + px[:,:,None]*right[None,None,:] + py[:,:,None]*up[None,None,:]).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs /= np.maximum(norms, 1e-6)
    return dirs

if NUMBA:
    @njit(cache=True, fastmath=True)
    def march(vol, bounds, origin, dirs, t_near, t_far, n_steps):
        nx, ny, nz = vol.shape
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        img_h, img_w, _ = dirs.shape
        img = np.zeros((img_h, img_w), dtype=np.float32)
        dt = (t_far - t_near) / max(n_steps-1, 1)
        out_counter_limit = 16
        for j in range(img_h):
            for i in range(img_w):
                out_count = 0
                acc = 0.0
                for k in range(n_steps):
                    t = t_near + k*dt
                    px = origin[0] + t*dirs[j,i,0]
                    py = origin[1] + t*dirs[j,i,1]
                    pz = origin[2] + t*dirs[j,i,2]
                    if (px < x_min) or (px > x_max) or (py < y_min) or (py > y_max) or (pz < z_min) or (pz > z_max):
                        out_count += 1
                        if out_count > out_counter_limit:
                            break
                        continue
                    out_count = 0
                    fx = (px - x_min) / (x_max - x_min) * (nx - 1)
                    fy = (py - y_min) / (y_max - y_min) * (ny - 1)
                    fz = (pz - z_min) / (z_max - z_min) * (nz - 1)
                    x0 = int(np.floor(fx)); x1 = x0+1
                    y0 = int(np.floor(fy)); y1 = y0+1
                    z0 = int(np.floor(fz)); z1 = z0+1
                    if (x0<0) or (x1>=nx) or (y0<0) or (y1>=ny) or (z0<0) or (z1>=nz):
                        continue
                    xd = fx - x0; yd = fy - y0; zd = fz - z0
                    C000 = vol[x0, y0, z0]; C100 = vol[x1, y0, z0]
                    C010 = vol[x0, y1, z0]; C110 = vol[x1, y1, z0]
                    C001 = vol[x0, y0, z1]; C101 = vol[x1, y0, z1]
                    C011 = vol[x0, y1, z1]; C111 = vol[x1, y1, z1]
                    C00 = C000*(1-xd) + C100*xd
                    C10 = C010*(1-xd) + C110*xd
                    C01 = C001*(1-xd) + C101*xd
                    C11 = C011*(1-xd) + C111*xd
                    C0 = C00*(1-yd) + C10*yd
                    C1 = C01*(1-yd) + C11*yd
                    C = C0*(1-zd) + C1*zd
                    acc += C * dt
                img[j,i] = acc
        return img
else:
    def march(vol, bounds, origin, dirs, t_near, t_far, n_steps):
        nx, ny, nz = vol.shape
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        img_h, img_w, _ = dirs.shape
        img = np.zeros((img_h, img_w), dtype=np.float32)
        dt = (t_far - t_near) / max(n_steps-1, 1)
        out_counter_limit = 16
        for j in range(img_h):
            for i in range(img_w):
                out_count = 0
                acc = 0.0
                for k in range(n_steps):
                    t = t_near + k*dt
                    px = origin[0] + t*dirs[j,i,0]
                    py = origin[1] + t*dirs[j,i,1]
                    pz = origin[2] + t*dirs[j,i,2]
                    if (px < x_min) or (px > x_max) or (py < y_min) or (py > y_max) or (pz < z_min) or (pz > z_max):
                        out_count += 1
                        if out_count > out_counter_limit:
                            break
                        continue
                    out_count = 0
                    fx = (px - x_min) / (x_max - x_min) * (nx - 1)
                    fy = (py - y_min) / (y_max - y_min) * (ny - 1)
                    fz = (pz - z_min) / (z_max - z_min) * (nz - 1)
                    x0 = int(np.floor(fx)); x1 = x0+1
                    y0 = int(np.floor(fy)); y1 = y0+1
                    z0 = int(np.floor(fz)); z1 = z0+1
                    if (x0<0) or (x1>=nx) or (y0<0) or (y1>=ny) or (z0<0) or (z1>=nz):
                        continue
                    xd = fx - x0; yd = fy - y0; zd = fz - z0
                    C000 = vol[x0, y0, z0]; C100 = vol[x1, y0, z0]
                    C010 = vol[x0, y1, z0]; C110 = vol[x1, y1, z0]
                    C001 = vol[x0, y0, z1]; C101 = vol[x1, y0, z1]
                    C011 = vol[x0, y1, z1]; C111 = vol[x1, y1, z1]
                    C00 = C000*(1-xd) + C100*xd
                    C10 = C010*(1-xd) + C110*xd
                    C01 = C001*(1-xd) + C101*xd
                    C11 = C011*(1-xd) + C111*xd
                    C0 = C00*(1-yd) + C10*yd
                    C1 = C01*(1-yd) + C11*yd
                    C = C0*(1-zd) + C1*zd
                    acc += C * dt
                img[j,i] = acc
        return img

class Renderer:
    def __init__(self):
        self.last_update = 0.0
        self.cache_key = None
        self.cached_dirs = None

    def camera(self, yaw_deg, pitch_deg):
        orbit_r = 140.0
        yaw = radians(float(yaw_deg))
        pitch = radians(float(pitch_deg))
        cx = 40.0 + orbit_r*np.cos(yaw)
        cy = orbit_r*np.sin(yaw)
        cz = 25.0
        cam_pos = np.array([cx, cy, cz], dtype=np.float32)
        target = np.array([60.0, 0.0, H], dtype=np.float32)
        right, up, fwd = look_at(cam_pos, target, np.array([0,0,1], dtype=np.float32))
        cr, sr = np.cos(radians(float(pitch_deg))), np.sin(radians(float(pitch_deg)))
        up2 =  cr*up - sr*fwd
        fwd2 = sr*up + cr*fwd
        return cam_pos, right, up2, fwd2

    def get_dirs(self, img_w, img_h, fov_deg, right, up, fwd):
        key = (img_w, img_h, float(fov_deg), float(right[0]), float(up[1]), float(fwd[2]))
        if self.cache_key != key or self.cached_dirs is None:
            self.cached_dirs = make_dirs(img_w, img_h, fov_deg, right, up, fwd)
            self.cache_key = key
        return self.cached_dirs

    def render(self, yaw, pitch, fov, img_w, img_h, steps):
        cam_pos, right, up, fwd = self.camera(yaw, pitch)
        dirs = self.get_dirs(img_w, img_h, fov, right, up, fwd)
        return march(VOL, BOUNDS, cam_pos, dirs, 0.0, 300.0, steps)

def robust_clim(img):
    finite = np.isfinite(img)
    if not np.any(finite):
        return 0.0, 1.0
    vals = img[finite]
    vmax = np.percentile(vals, 99.0)
    vmin = np.percentile(vals, 1.0)
    if vmax <= vmin:
        vmax = np.max(vals)
        vmin = np.min(vals)
        if vmax == vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)

R = Renderer()

def main():
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    plt.subplots_adjust(left=0.12, bottom=0.28, top=0.92)
    img = R.render(60, 0, 45, 360, 360, 220)
    vmin, vmax = robust_clim(img)
    im = ax.imshow(img, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title('ppm·m (fast sliders; robust autoscale)')
    ax.set_axis_off()

    ax_yaw   = plt.axes([0.12, 0.20, 0.74, 0.03])
    ax_pitch = plt.axes([0.12, 0.16, 0.74, 0.03])
    ax_fov   = plt.axes([0.12, 0.12, 0.74, 0.03])
    ax_res   = plt.axes([0.12, 0.08, 0.74, 0.03])
    ax_steps = plt.axes([0.12, 0.04, 0.74, 0.03])

    s_yaw   = Slider(ax_yaw,   'Yaw',   0, 180, valinit=60, valstep=1)
    s_pitch = Slider(ax_pitch, 'Pitch', -45, 45, valinit=0,  valstep=1)
    s_fov   = Slider(ax_fov,   'FOV',   15, 90, valinit=45,  valstep=1)
    s_res   = Slider(ax_res,   'Img',   160, 720, valinit=360, valstep=40)
    s_steps = Slider(ax_steps, 'Steps', 60, 600, valinit=220, valstep=10)

    ax_btn = plt.axes([0.80, 0.93, 0.16, 0.05])
    btn = Button(ax_btn, 'Save PNG')

    MIN_DT = 0.05
    FAST_RES = 200
    FAST_STEPS = 120
    dragging = {'active': False}

    def redraw(full=False):
        res = int(s_res.val if full else FAST_RES)
        steps = int(s_steps.val if full else FAST_STEPS)
        img2 = R.render(s_yaw.val, s_pitch.val, s_fov.val, res, res, steps)
        img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = robust_clim(img2)
        im.set_data(img2)
        im.set_clim(vmin, vmax)
        im.set_extent((0, img2.shape[1], 0, img2.shape[0]))
        fig.canvas.draw_idle()

    def throttled_update(val=None):
        now = time.time()
        if now - R.last_update < MIN_DT:
            return
        R.last_update = now
        if dragging['active']:
            redraw(full=False)
        else:
            redraw(full=True)

    def on_press(event):
        if event.inaxes in (ax_yaw, ax_pitch, ax_fov, ax_res, ax_steps):
            dragging['active'] = True
            redraw(full=False)

    def on_release(event):
        if dragging['active']:
            dragging['active'] = False
            redraw(full=True)

    def save(evt):
        arr = im.get_array()
        plt.imsave('ppm_m_view.png', arr, origin='lower', cmap='viridis')
        print('Saved: ppm_m_view.png')

    for s in (s_yaw, s_pitch, s_fov, s_res, s_steps):
        s.on_changed(throttled_update)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    btn.on_clicked(save)

    plt.show()

if __name__ == '__main__':
    main()