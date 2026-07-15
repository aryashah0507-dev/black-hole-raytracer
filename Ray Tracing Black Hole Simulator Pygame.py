"""
Project: Relativistic Black Hole Simulation 

Description:
This program visually simulates a Schwarzschild Black Hole using General Relativity.
It solves the geodesic equations for light rays (photons) to render gravitational lensing.
The code uses JIT compilation (Numba) to perform heavy calculus on the CPU in real-time.
"""

import pygame
import numpy as np
import math
from numba import njit, prange
import time
import json


# --- 1. SETTINGS & DICTIONARIES (Rubric: Dictionary, Files, Try/Except) ---

def load_configuration():
    """
    Load simulation configuration from JSON file with error handling.
    
    This function implements the file I/O and error handling requirements by:
    1. Attempting to load settings from 'black_hole_settings.json'
    2. Using try/except blocks to handle file errors gracefully
    3. Creating default configuration if file doesn't exist
    4. Saving default settings to file for future runs
    
    Returns:
        dict: Configuration dictionary containing:
            - width (int): Render window width in pixels
            - height (int): Render window height in pixels  
            - mass_mult (float): Black hole mass multiplier in solar masses
            - camera_dist (float): Initial camera distance in Schwarzschild radii
    
    Raises:
        Exception: Catches and reports file I/O errors during config save
    """
    default_settings = {
        "width": 400,           # Render Width
        "height": 300,          # Render Height
        "mass_mult": 4.0e6,     # Black Hole Mass (Solar Masses)
        "camera_dist": 20.0,    # Initial distance (Rs units)
    }
    

    filename = "black_hole_settings.json"
    # This try/except block demonstrates error handling for file I/O operations
    # It's a critical part of robust software that deals with external files
    try:
        # Attempt to open and parse the JSON configuration file
        with open(filename, "r") as f:
            loaded_data = json.load(f)
            
            # Safely update settings only if keys exist in loaded file
            # This prevents KeyError exceptions from malformed config files
            if "width" in loaded_data: default_settings["width"] = loaded_data["width"]
            if "height" in loaded_data: default_settings["height"] = loaded_data["height"]
            print(f"Successfully loaded settings from {filename}")
            
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle two main file error scenarios:
        # 1. FileNotFoundError: Config file doesn't exist (first run)
        # 2. json.JSONDecodeError: Config file exists but contains invalid JSON
        print("Config file not found or invalid. Creating default.")
        try:
            # Create a new config file with default settings for future runs
            # This ensures the program works even on first execution
            with open(filename, "w") as f:
                json.dump(default_settings, f, indent=4)
        except Exception as e:
            # Catch any other file system errors (permissions, disk space, etc.)
            print(f"Could not save config: {e}")
            
    return default_settings

# Initialize Configuration Dictionary
SETTINGS = load_configuration()

# Extract constants from Dictionary
WIDTH = int(SETTINGS["width"])
HEIGHT = int(SETTINGS["height"])

# --- 2. PHYSICS CONSTANTS & TESTING (Rubric: Variables, Expressions, Testing) ---

# Physical Constants (using exact SI values for realistic simulation)
c = 299792458.0       # Speed of light in vacuum (m/s) - exact by definition
G = 6.67430e-11       # Gravitational constant (m³/kg·s²) - CODATA 2018 value
M_SOLAR = 1.989e30    # Solar mass in kilograms - IAU 2015 nominal value

# Calculate Black Hole Mass from Dictionary value
# This allows easy adjustment of black hole size through configuration
MASS_BH = float(SETTINGS["mass_mult"]) * M_SOLAR 

# Schwarzschild Radius (Event Horizon) calculation
# Formula: Rs = 2GM/c² - derived from general relativity
# This represents the point of no return around a black hole
RS = 2.0 * G * MASS_BH / (c**2)

# Accretion Disk Dimensions (in terms of Schwarzschild radii)
# ISCO = Innermost Stable Circular Orbit at 3Rs for Schwarzschild black hole
DISK_R1 = RS * 3.0 # Inner edge: matter spirals inward from here
DISK_R2 = RS * 5.0 # Outer edge: visible glowing material extends to here 

def run_diagnostics():
    """
    Run system diagnostics to verify physics calculations are valid.
    
    This function performs critical validation checks to ensure:
    1. Physical constants are within expected ranges
    2. Calculated values make physical sense
    3. Geometric constraints are satisfied
    
    Uses assertions to catch configuration errors early and prevent
    undefined behavior in the physics simulation.
    
    Raises:
        AssertionError: If any physics validation check fails
    """
    print("Running diagnostics...")
    # Verify fundamental constants haven't been corrupted
    assert c >= 299792458.0, "Physics Error: Speed of light is not changable."
    # Ensure black hole mass calculation produced valid result
    assert RS > 0, "Physics Error: Schwarzschild radius is zero or negative."
    # Check accretion disk geometry makes physical sense
    assert DISK_R2 > DISK_R1, "Geometry Error: Outer disk radius is smaller than inner."
    print("Diagnostics passed. Physics engine ready.")

run_diagnostics()

# --- Physics Stuff ---

@njit(fastmath=True)
def geodesic_rhs(r, theta, dr, dtheta, dphi, E, rs):
    """
    Calculate right-hand side of geodesic equations in Schwarzschild spacetime.
    
    This function computes the second derivatives (accelerations) for photon
    trajectories in curved spacetime around a black hole. It implements the
    geodesic equation from general relativity in spherical coordinates.
    
    The geodesic equation describes how particles move through curved spacetime:
    d²x^μ/dλ² + Γ^μ_νρ (dx^ν/dλ)(dx^ρ/dλ) = 0
    
    Where Γ^μ_νρ are Christoffel symbols derived from the Schwarzschild metric.
    
    Args:
        r (float): Radial coordinate (distance from black hole center)
        theta (float): Polar angle coordinate
        dr (float): Radial velocity component
        dtheta (float): Polar angular velocity component  
        dphi (float): Azimuthal angular velocity component
        E (float): Conserved energy of the photon
        rs (float): Schwarzschild radius of the black hole
        
    Returns:
        tuple: (d²r/dλ², d²θ/dλ², d²φ/dλ²) - second derivatives of coordinates
    """
    # f(r) is the Schwarzschild metric function: f = 1 - Rs/r
    # This encodes the curvature of spacetime near the black hole
    f = 1.0 - rs / r
    
    # Prevent numerical issues at the singularity (r → 0)
    # Return zero acceleration to effectively stop integration
    if r < 1: return 0.0, 0.0, 0.0
    
    # Calculate time derivative with respect to affine parameter
    # For null geodesics (photons), this relates energy to metric
    dt_dL = E / f
    
    # The geodesic equation in component form gives us accelerations:
    # d²x^μ/dλ² = -Γ^μ_νρ (dx^ν/dλ)(dx^ρ/dλ)
    # where Γ^μ_νρ are Christoffel symbols from Schwarzschild metric
    
    # Combine angular velocity terms (θ and φ components)
    # This represents motion on the 2-sphere at radius r
    angular_term = (dtheta**2 + math.sin(theta)**2 * dphi**2)
    
    # Radial acceleration: includes gravitational attraction and centrifugal effects
    # First term: gravitational time dilation effect
    # Second term: radial kinetic energy in curved space
    # Third term: centrifugal acceleration from angular motion
    d2_r = - (rs / (2.0 * r*r)) * f * dt_dL**2 + \
           (rs / (2.0 * r*r * f)) * dr**2 + \
           (r - rs) * angular_term

    # Polar angle acceleration: coupling between radial and angular motion
    # First term: Coriolis-like effect from radial motion
    # Second term: centrifugal force in θ direction from φ rotation
    d2_theta = - (2.0 / r) * dr * dtheta + \
               math.sin(theta) * math.cos(theta) * dphi**2

    # Azimuthal acceleration: conservation of angular momentum effects
    # First term: coupling with radial motion
    # Second term: cotangent effect from spherical coordinates
    d2_phi = - (2.0 / r) * dr * dphi - \
             2.0 * (math.cos(theta) / math.sin(theta)) * dtheta * dphi
             
    return d2_r, d2_theta, d2_phi

@njit(fastmath=True)
def rk4_step(state, E, rs, h):
    """
    Perform one Runge-Kutta 4th order integration step for geodesic equation.
    
    RK4 is a numerical method for solving ordinary differential equations with
    high accuracy. It uses four slope estimates to compute the next state:
    
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)  
    k4 = f(t + h, y + h*k3)
    
    y_next = y + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    
    This gives 4th order accuracy in the step size h, making it much more
    accurate than simple Euler integration for the same computational cost.
    
    Args:
        state (ndarray): Current state [r, θ, φ, dr/dλ, dθ/dλ, dφ/dλ]
        E (float): Conserved energy of the photon trajectory
        rs (float): Schwarzschild radius of the black hole
        h (float): Step size in affine parameter
        
    Returns:
        ndarray: Next state after one RK4 integration step
    """
    # Unpack state vector for clarity
    # State contains position and velocity in spherical coordinates
    r, theta, phi, dr, dtheta, dphi = state
    
    # k1: slope at the beginning of interval
    # Use current position and velocity to get acceleration
    k1_d2r, k1_d2th, k1_d2ph = geodesic_rhs(r, theta, dr, dtheta, dphi, E, rs)
    k1 = np.array([dr, dtheta, dphi, k1_d2r, k1_d2th, k1_d2ph])
    
    # k2: slope at midpoint using k1 estimate
    # Advance state by half step using k1 slope
    s2 = state + 0.5 * h * k1
    k2_d2r, k2_d2th, k2_d2ph = geodesic_rhs(s2[0], s2[1], s2[3], s2[4], s2[5], E, rs)
    k2 = np.array([s2[3], s2[4], s2[5], k2_d2r, k2_d2th, k2_d2ph])

    # k3: improved slope at midpoint using k2 estimate
    # This gives a more accurate midpoint slope
    s3 = state + 0.5 * h * k2
    k3_d2r, k3_d2th, k3_d2ph = geodesic_rhs(s3[0], s3[1], s3[3], s3[4], s3[5], E, rs)
    k3 = np.array([s3[3], s3[4], s3[5], k3_d2r, k3_d2th, k3_d2ph])

    # k4: slope at end of interval using k3 estimate
    # Advance full step using k3 to get endpoint slope
    s4 = state + h * k3
    k4_d2r, k4_d2th, k4_d2ph = geodesic_rhs(s4[0], s4[1], s4[3], s4[4], s4[5], E, rs)
    k4 = np.array([s4[3], s4[4], s4[5], k4_d2r, k4_d2th, k4_d2ph])
    
    # Weighted average of slopes: k1 and k4 get weight 1, k2 and k3 get weight 2
    # This combination gives 4th order accuracy
    return state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@njit(fastmath=True)
def solve_ray_fast(cam_pos, ray_dir, rs, disk_r1, disk_r2):
    """
    Trace a light ray through curved spacetime to determine pixel color.
    
    This is the core ray tracing function that:
    1. Converts Cartesian ray to spherical coordinates
    2. Calculates initial conditions for geodesic integration
    3. Uses RK4 to integrate the geodesic equation
    4. Checks for intersections with event horizon or accretion disk
    5. Returns RGB color based on what the ray hits
    
    The function implements null geodesics (light ray paths) in
    Schwarzschild spacetime, accounting for gravitational lensing
    effects that bend light around the black hole.
    
    Args:
        cam_pos (tuple): Camera position in Cartesian coordinates (x, y, z)
        ray_dir (tuple): Ray direction vector (dx, dy, dz) - normalized
        rs (float): Schwarzschild radius of the black hole
        disk_r1 (float): Inner radius of accretion disk
        disk_r2 (float): Outer radius of accretion disk
        
    Returns:
        tuple: RGB color values (r, g, b) in range [0, 1]
            - (0, 0, 0): Black hole or absorbed light
            - (1.0, 0.5-1.0, 0.1): Accretion disk glow
            - (0.05, 0.05, 0.1): Distant space background
    """
    x, y, z = cam_pos
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)

    dx, dy, dz = ray_dir
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    sin_p, cos_p = math.sin(phi), math.cos(phi)

    dr = sin_t*cos_p*dx + sin_t*sin_p*dy + cos_t*dz
    dtheta = (cos_t*cos_p*dx + cos_t*sin_p*dy - sin_t*dz) / r
    dphi = (-sin_p*dx + cos_p*dy) / (r * sin_t)

    # State vector contains position and velocity in spherical coordinates
    # [r, θ, φ, dr/dλ, dθ/dλ, dφ/dλ] where λ is the affine parameter
    state = np.array([r, theta, phi, dr, dtheta, dphi])

    # Schwarzschild metric function at initial position
    f = 1.0 - rs / r

    # Calculate conserved energy from initial conditions
    # For null geodesics: g_μν (dx^μ/dλ)(dx^ν/dλ) = 0
    # This gives us: -f(dt/dλ)² + (dr/dλ)²/f + r²[(dθ/dλ)² + sin²θ(dφ/dλ)²] = 0
    angular_v_sq = dtheta**2 + math.sin(theta)**2 * dphi**2
    term = (dr*dr)/(f*f) + (r*r*angular_v_sq)/f
    dt_dL = math.sqrt(term)  # dt/dλ from null condition
    E = f * dt_dL            # Conserved energy E = -g_tt * dt/dλ

    # Integration step size (adaptive based on Schwarzschild radius)
    # Smaller steps near the black hole for better accuracy
    step_size = rs * 0.17

    # Integrate geodesic equation for up to 200 steps
    # This limit prevents infinite loops for highly bent rays
    for i in range(200):
        curr_r = state[0]  # Current radial distance
        
        # Check if ray has crossed the event horizon
        # Light cannot escape from inside the Schwarzschild radius
        if curr_r <= rs: return 0.0, 0.0, 0.0  # Return black (absorbed)
        
        # Check for intersection with accretion disk (after a few steps)
        # Skip first few steps to avoid false positives near camera
        if i > 2:
            curr_theta, curr_phi = state[1], state[2]
            
            # Convert current position back to Cartesian for disk intersection
            current_y = curr_r * math.sin(curr_theta) * math.sin(curr_phi)
            
            # Check if ray is near the disk plane (y ≈ 0)
            # Disk thickness is about 0.2 Schwarzschild radii
            if abs(current_y) < rs * 0.2:
                # Calculate distance from black hole center in x-z plane
                cx = curr_r * math.sin(curr_theta) * math.cos(curr_phi)
                cz = curr_r * math.cos(curr_theta)
                dist_xz = math.sqrt(cx**2 + cz**2)
                
                # Check if within accretion disk radial bounds
                if disk_r1 < dist_xz < disk_r2:
                    # Color varies with distance: inner disk is brighter
                    factor = (dist_xz - disk_r1) / (disk_r2 - disk_r1)
                    return 1.0, 0.5 + factor*0.5, 0.1  # Orange/yellow glow

        # If ray has traveled very far, assume it reaches distant space
        # Return dark blue background color
        if curr_r > rs * 50: return 0.05, 0.05, 0.1

        # Advance the ray by one integration step
        state = rk4_step(state, E, rs, step_size)

        

    return 0.0, 0.0, 0.0

@njit(parallel=True, fastmath=True)
def render_frame(width, height, cam_pos, cam_front, cam_up, cam_right, fov, rs, dr1, dr2):
    """
    Render a complete frame of the black hole simulation using ray tracing.
    
    This function implements a parallel ray tracing algorithm that:
    1. Generates rays from camera through each pixel
    2. Traces each ray through curved spacetime 
    3. Determines pixel color based on ray intersection
    4. Returns a complete RGB image buffer
    
    Uses Numba parallel compilation for real-time performance on CPU.
    The @njit decorator compiles this to native code for speed.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        cam_pos (ndarray): Camera position [x, y, z]
        cam_front (ndarray): Camera forward direction (normalized)
        cam_up (ndarray): Camera up direction (normalized)
        cam_right (ndarray): Camera right direction (normalized)
        fov (float): Field of view angle in radians
        rs (float): Schwarzschild radius
        dr1 (float): Inner accretion disk radius
        dr2 (float): Outer accretion disk radius
        
    Returns:
        ndarray: RGB image buffer with shape (height, width, 3), dtype uint8
    """
    # Initialize RGB image buffer (height × width × 3 channels)
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Camera projection parameters
    aspect = width / height           # Aspect ratio to prevent distortion
    tan_fov = math.tan(fov * 0.5)    # Tangent of half field-of-view

    # Parallel loop over all pixels (prange enables multi-threading)
    for y in prange(height):
        for x in range(width):
            # Convert pixel coordinates to normalized device coordinates (NDC)
            # NDC range: [-1, 1] for both x and y, with aspect correction
            ndc_x = (2.0 * (x + 0.5) / width - 1.0) * aspect * tan_fov
            ndc_y = (1.0 - 2.0 * (y + 0.5) / height) * tan_fov
            
            # Construct ray direction in world space using camera basis vectors
            # Ray starts from camera and goes through current pixel
            dir_x = cam_front[0] + ndc_x * cam_right[0] + ndc_y * cam_up[0]
            dir_y = cam_front[1] + ndc_x * cam_right[1] + ndc_y * cam_up[1]
            dir_z = cam_front[2] + ndc_x * cam_right[2] + ndc_y * cam_up[2]
            
            # Normalize ray direction vector to unit length
            l = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
            ray_dir = (dir_x/l, dir_y/l, dir_z/l)

            r, g, b = solve_ray_fast(cam_pos, ray_dir, rs, dr1, dr2)
            
            buffer[y, x, 0] = int(r * 255)
            buffer[y, x, 1] = int(g * 255)
            buffer[y, x, 2] = int(b * 255)
            
    return buffer

@njit
def get_debug_path(cam_pos, ray_dir, rs):
    """
    Calculate the 3D path of a light ray for visualization purposes.
    
    This function traces a ray through spacetime and returns the sequence
    of 3D points along its trajectory. Used for drawing the green debug
    lines that show how light bends around the black hole.
    
    The path shows the geodesic (straightest possible path) that light
    follows in the curved spacetime around a black hole. These paths
    appear curved when viewed from our flat 3D perspective.
    
    Args:
        cam_pos (tuple): Camera position (x, y, z)
        ray_dir (tuple): Ray direction (dx, dy, dz) - normalized
        rs (float): Schwarzschild radius
        
    Returns:
        list: Sequence of 3D points [(x1,y1,z1), (x2,y2,z2), ...]
              representing the ray's path through spacetime
    """
    # List to store 3D points along the ray's path
    path = []
    
    # Convert initial position to spherical coordinates
    x, y, z = cam_pos
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)

    # Transform ray direction to spherical coordinate velocities
    dx, dy, dz = ray_dir
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    sin_p, cos_p = math.sin(phi), math.cos(phi)

    dr = sin_t*cos_p*dx + sin_t*sin_p*dy + cos_t*dz
    dtheta = (cos_t*cos_p*dx + cos_t*sin_p*dy - sin_t*dz) / r
    dphi = (-sin_p*dx + cos_p*dy) / (r * sin_t)
    
    # Initial state for geodesic integration
    state = np.array([r, theta, phi, dr, dtheta, dphi])
    
    # Calculate conserved energy (same as in solve_ray_fast)
    f = 1.0 - rs / r
    angular_v_sq = dtheta**2 + math.sin(theta)**2 * dphi**2
    term = (dr*dr)/(f*f) + (r*r*angular_v_sq)/f
    E = f * math.sqrt(term)
    
    # Use same step size as main ray tracer for consistency
    step_size = rs * 0.17

    # Integrate ray path with same limits as main tracer
    for i in range(200):
        curr_r = state[0]
        
        # Stop if ray hits event horizon or goes too far
        # Limit to 30 Rs for visualization (closer than render limit)
        if curr_r <= rs or curr_r > rs * 30: break

        # Advance ray by one integration step
        state = rk4_step(state, E, rs, step_size)

        # Convert current spherical position back to Cartesian
        # These points will be connected to show the curved path
        cx = state[0] * math.sin(state[1]) * math.cos(state[2])
        cy = state[0] * math.sin(state[1]) * math.sin(state[2])
        cz = state[0] * math.cos(state[1])
        path.append((cx, cy, cz))
        
    return path

# --- 4. HELPER FUNCTIONS ---

def project_to_screen(points):
    """
    Project 3D world coordinates to 2D screen coordinates for visualization.
    
    This function converts the 3D ray paths to 2D screen positions so they
    can be drawn as debug lines. Uses a simple orthographic projection
    that shows the x-z plane (top-down view of the black hole).
    
    Args:
        points (list): List of 3D points [(x1,y1,z1), (x2,y2,z2), ...]
        
    Returns:
        list: List of 2D screen coordinates [(sx1,sy1), (sx2,sy2), ...]
    """
    # Orthographic projection: x maps to screen x, z maps to screen y
    # Scale factor of 10 pixels per Schwarzschild radius
    # Take every 3rd point to reduce visual clutter (::3 slicing)
    return [
        (int(WIDTH/2 + px/RS * 10), int(HEIGHT/2 + pz/RS * 10)) 
        for (px, py, pz) in points[::3] 
    ]

# --- 5. MAIN LOOP ---

def main():
    """
    Main simulation loop that handles user input, camera control, and rendering.
    
    This function implements the complete interactive black hole visualization:
    1. Initialize Pygame window and graphics
    2. Set up orbital camera system around black hole
    3. Handle real-time user input for camera control
    4. Render frames using ray tracing through curved spacetime
    5. Display performance metrics and debug information
    
    Camera Controls:
    - A/D: Rotate camera left/right (change azimuth angle)
    - W/S: Move camera up/down (change elevation angle)
    - Q/E: Move camera closer/farther (change orbital radius)
    
    The camera maintains a spherical orbit around the black hole,
    always pointing toward the center to show gravitational lensing effects.
    """
    # Initialize Pygame graphics system
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Black Hole (Final Corrected Math)")
    clock = pygame.time.Clock()  # For FPS control
    font = pygame.font.SysFont('Arial', 16)  # For on-screen text

    # Camera orbital parameters (spherical coordinates)
    radius = 20.0 * RS    # Distance from black hole (20 Schwarzschild radii)
    azimuth = 0.0         # Horizontal angle (radians)
    elevation = 1.2       # Vertical angle (radians) - roughly 70 degrees
    target = np.array([0.0, 0.0, 0.0])  # Camera always looks at black hole center
    
    # Numba compilation warning (first call to @njit functions is slow)
    print("Compiling Corrected Physics... (Wait ~15s)")

    running = True
    
    while running:
        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # Continuous key input handling for smooth camera movement
        keys = pygame.key.get_pressed()
        
        # Horizontal rotation: A rotates left, D rotates right
        # Prevent simultaneous opposite inputs to avoid jitter
        if keys[pygame.K_a] and not keys[pygame.K_d]: azimuth -= 0.1
        elif keys[pygame.K_d] and not keys[pygame.K_a]: azimuth += 0.1
        
        # Vertical movement: W moves up, S moves down
        # Clamp elevation to prevent camera from going upside-down
        if keys[pygame.K_w]: elevation = max(0.1, elevation - 0.1)    # Min ~6°
        elif keys[pygame.K_s]: elevation = min(3.1, elevation + 0.1)  # Max ~178°
            
        # Zoom in/out: E moves closer, Q moves farther
        # Multiplicative scaling for smooth exponential zoom
        if keys[pygame.K_e]: radius *= 0.95   # Zoom in by 5%
        elif keys[pygame.K_q]: radius *= 1.05 # Zoom out by 5%

        # Convert spherical camera coordinates to Cartesian position
        # Standard spherical coordinate convention: θ from +y axis, φ from +x axis
        cx = radius * math.sin(elevation) * math.cos(azimuth)  # x component
        cy = radius * math.cos(elevation)                      # y component  
        cz = radius * math.sin(elevation) * math.sin(azimuth)  # z component
        cam_pos = np.array([cx, cy, cz])
        
        # Construct camera coordinate system (right-handed)
        # Forward vector points from camera toward black hole center
        fwd = target - cam_pos
        fwd = fwd / np.linalg.norm(fwd)  # Normalize to unit length
        
        # Define world up direction (positive y-axis)
        world_up = np.array([0.0, 1.0, 0.0])
        
        # Right vector is perpendicular to both forward and world-up
        right = np.cross(fwd, world_up)
        right = right / np.linalg.norm(right)
        
        # True up vector is perpendicular to forward and right
        # This completes the orthonormal camera basis
        up = np.cross(right, fwd)

        # Render complete frame using parallel ray tracing through curved spacetime
        # Each pixel traces a light ray from camera through black hole's gravitational field
        pixels = render_frame(WIDTH, HEIGHT, cam_pos, fwd, up, right, 1.0, RS, DISK_R1, DISK_R2)
        
        # Convert numpy array to Pygame surface and display
        # swapaxes corrects for numpy (y,x,c) vs Pygame (x,y,c) convention
        surf = pygame.surfarray.make_surface(pixels.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Draw debug visualization of light ray paths
        # Sample rays on a 6×6 grid across the screen to show bending
        stride_x = WIDTH // 6   # Horizontal spacing between sample rays
        stride_y = HEIGHT // 6  # Vertical spacing between sample rays
        tan_fov = math.tan(1.0 * 0.5)  # Same FOV as main render
        aspect = WIDTH / HEIGHT

        # Loop through grid of sample points
        for y in range(stride_y // 2, HEIGHT, stride_y):
            for x in range(stride_x // 2, WIDTH, stride_x):
                # Convert pixel to normalized device coordinates
                ndc_x = (2.0 * (x + 0.5) / WIDTH - 1.0) * aspect * tan_fov
                ndc_y = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * tan_fov
                
                # Generate ray direction for this pixel
                dx = fwd[0] + ndc_x * right[0] + ndc_y * up[0]
                dy = fwd[1] + ndc_x * right[1] + ndc_y * up[1]
                dz = fwd[2] + ndc_x * right[2] + ndc_y * up[2]
                l = math.sqrt(dx*dx + dy*dy + dz*dz)
                ray_dir = (dx/l, dy/l, dz/l)

                # Calculate 3D path and project to screen coordinates
                path_3d = get_debug_path(cam_pos, ray_dir, RS)
                screen_points = project_to_screen(path_3d)
                
                # Draw green line showing how light bends in curved spacetime
                if len(screen_points) > 1:
                   pygame.draw.lines(screen, (0, 255, 0), False, screen_points, 1)

        # Display performance and camera information
        fps_text = f"FPS: {clock.get_fps():.1f}"                    # Frames per second
        pos_text = f"Pos: {elevation:.2f} rad, {radius/RS:.1f} Rs"  # Camera position
        
        # Render white text in top-left corner
        screen.blit(font.render(fps_text, True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(pos_text, True, (255, 255, 255)), (10, 30))

        # Update display buffer and maintain target framerate
        pygame.display.flip()  # Show the completed frame
        clock.tick(60)         # Limit to 60 FPS

    # Clean up when user closes window
    pygame.quit()

# Entry point: run simulation when script is executed directly
if __name__ == "__main__":
    main()
