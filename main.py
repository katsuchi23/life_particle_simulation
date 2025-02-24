import pygame
import numpy as np
import cupy as cp
import math
import pygame_gui

# -- SIMULATION PARAMETERS --
PLAYGROUND_WIDTH = 1000
Y_SIZE = 1000
CONTROL_WIDTH = 200
TOTAL_WIDTH = PLAYGROUND_WIDTH + CONTROL_WIDTH
DRAW_INTERVAL = 1       # Draw every simulation step
FPS = 60               # Frames per second
GRAVITY = 0.1          # Gravity constant

# --- Initial Settings for Atom Types ---
# For each type, we store:
#   - name: used for labeling the UI.
#   - color: the drawing color.
#   - num_particles: how many particles of that type.
#   - radius: drawing radius (and collision diameter = 2*radius).
INITIAL_ATOM_SETTINGS = [
    {"name": "red",    "color": (255, 0, 0),   "num_particles": 2000, "radius": 3.0},
    {"name": "white", "color": (255, 255, 255), "num_particles": 2000, "radius": 3.0},
    {"name": "blue",   "color": (0, 0, 255),   "num_particles": 2000, "radius": 3.0},
    {"name": "green",  "color": (0, 255, 0),   "num_particles": 2000, "radius": 3.0},
]
NUM_TYPES = len(INITIAL_ATOM_SETTINGS)

# --- Initial Force Matrix ---
# This is a NUM_TYPES x NUM_TYPES matrix.
INITIAL_FORCE_MATRIX = np.array([
    [1,  1,  1,  1],
    [1,  1,  1,  1],
    [1,  1,  1,  1],
    [1,  1,  1,  1]
], dtype=np.float32)


class GPUParticleSimulation:
    # ---------------------------
    # CUDA Kernel: Compute Forces
    # ---------------------------
    force_kernel_code = r'''
extern "C" __global__
void compute_forces_kernel(float* positions, float* velocities, const int* types,
                             const float* force_matrix, int n_particles, int num_types, float interaction_radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;
    
    float fx = 0.0f;
    float fy = 0.0f;
    
    float px = positions[idx * 2];
    float py = positions[idx * 2 + 1];
    int p_type = types[idx];
    
    for (int j = 0; j < n_particles; j++) {
        if (j == idx) continue;
        float dx = positions[j * 2] - px;
        float dy = positions[j * 2 + 1] - py;
        float distance = sqrtf(dx * dx + dy * dy);
        if (distance > 0.0f && distance < interaction_radius) {
            int other_type = types[j];
            float force = force_matrix[p_type * num_types + other_type] / distance;
            fx += force * dx;
            fy += force * dy;
        }
    }
    velocities[idx * 2]     = (velocities[idx * 2]     + fx) * 0.5f;
    velocities[idx * 2 + 1] = (velocities[idx * 2 + 1] + fy) * 0.5f;
}
'''

    # -------------------------------
    # CUDA Kernel: Resolve Collisions
    # -------------------------------
    collision_kernel_code = r'''
extern "C" __global__
void resolve_collisions_kernel(float* positions, float* velocities,
                               int n_particles, float* radii) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    float ri = radii[i];
    
    for (int j = i + 1; j < n_particles; j++) {
         float rj = radii[j];
         float desired = ri + rj;  // desired separation
         float dx = positions[j * 2] - positions[i * 2];
         float dy = positions[j * 2 + 1] - positions[i * 2 + 1];
         float dist = sqrtf(dx * dx + dy * dy);
         if (dist > 0.0f && dist < desired) {
             float overlap = desired - dist;
             float nx = dx / dist;
             float ny = dy / dist;
             float correctionX = nx * overlap * 0.5f;
             float correctionY = ny * overlap * 0.5f;
             
             atomicAdd(&positions[i * 2], -correctionX);
             atomicAdd(&positions[i * 2 + 1], -correctionY);
             atomicAdd(&positions[j * 2],  correctionX);
             atomicAdd(&positions[j * 2 + 1],  correctionY);
             
             float rvx = velocities[i * 2] - velocities[j * 2];
             float rvy = velocities[i * 2 + 1] - velocities[j * 2 + 1];
             float relVel = rvx * nx + rvy * ny;
             
             if (relVel < 0.0f) {
                 float impulse = -(2.0f * relVel) / 2.0f;
                 float impulseX = impulse * nx;
                 float impulseY = impulse * ny;
                 
                 atomicAdd(&velocities[i * 2], impulseX);
                 atomicAdd(&velocities[i * 2 + 1], impulseY);
                 atomicAdd(&velocities[j * 2], -impulseX);
                 atomicAdd(&velocities[j * 2 + 1], -impulseY);
             }
         }
    }
}
'''

    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((TOTAL_WIDTH, Y_SIZE))
        self.ui_manager = pygame_gui.UIManager((TOTAL_WIDTH, Y_SIZE))
        pygame.display.set_caption("GPU Particle Simulation with Separate Control Panel")
        
        # Copy initial settings.
        self.atom_params = [atom.copy() for atom in INITIAL_ATOM_SETTINGS]
        self.force_params = INITIAL_FORCE_MATRIX.copy()
        self.interaction_radius = 80.0
        self.fps = FPS
        self.gravity = GRAVITY
        
        self._create_ui_panel()
        self.initialize_simulation()
        
        self.compute_forces_kernel = cp.RawKernel(self.force_kernel_code, "compute_forces_kernel")
        self.resolve_collisions_kernel = cp.RawKernel(self.collision_kernel_code, "resolve_collisions_kernel")
        self.force_matrix = cp.array((self.force_params * self.gravity).flatten())

    def _create_ui_panel(self):
        """Create a scrolling control panel in the control area."""
        panel_rect = pygame.Rect(PLAYGROUND_WIDTH + 10, 10, CONTROL_WIDTH - 20, Y_SIZE - 20)
        
        # Calculate total height needed for all UI elements to set proper scrollable height
        element_height = 30
        padding = 5
        total_height = 0
        
        # Add height for FPS and Gravity controls
        simulation_params_height = (element_height + padding) * 3  # Title + FPS + Gravity
        
        # Base height for atom type settings
        atom_section_height = (element_height + padding) * 3  # Title + count + radius
        for _ in self.atom_params:
            total_height += atom_section_height + padding
            
        # Height for force matrix
        force_title_height = element_height + padding
        force_row_height = element_height + padding
        force_cell_height = (element_height + padding) * NUM_TYPES
        force_matrix_height = force_title_height + (force_row_height + force_cell_height) * NUM_TYPES + padding
        
        # Button height
        button_height = 40 + padding * 2
        
        total_height += simulation_params_height + force_matrix_height + button_height
        
        # Make sure there's enough content to force scrolling
        scrollable_height = max(total_height, Y_SIZE - 20)
        
        # Use a scrolling container with the calculated height
        self.ui_panel = pygame_gui.elements.UIScrollingContainer(
            relative_rect=panel_rect,
            manager=self.ui_manager
        )
        
        # Set the scrollable area height
        self.ui_panel.set_scrollable_area_dimensions((panel_rect.width - 20, scrollable_height))
        
        self.ui_elements = {}
        container_width = self.ui_panel.get_container().get_size()[0]
        y_offset = 10
        
        # Add simulation parameters section (FPS and Gravity)
        sim_title = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, container_width - 20, element_height),
                                              text="Simulation Settings:",
                                              manager=self.ui_manager,
                                              container=self.ui_panel.get_container())
        y_offset += element_height + padding
        
        fps_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, 60, element_height),
                                               text="FPS:",
                                               manager=self.ui_manager,
                                               container=self.ui_panel.get_container())
        fps_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(80, y_offset, container_width - 90, element_height),
                                                       manager=self.ui_manager,
                                                       container=self.ui_panel.get_container())
        fps_entry.set_text(str(self.fps))
        y_offset += element_height + padding
        
        gravity_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, 60, element_height),
                                                   text="Gravity:",
                                                   manager=self.ui_manager,
                                                   container=self.ui_panel.get_container())
        gravity_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(80, y_offset, container_width - 90, element_height),
                                                           manager=self.ui_manager,
                                                           container=self.ui_panel.get_container())
        gravity_entry.set_text(str(self.gravity))
        y_offset += element_height + padding
        
        # Store the UI elements for FPS and gravity
        self.ui_elements["fps"] = fps_entry
        self.ui_elements["gravity"] = gravity_entry
        
        # For each atom type, add UI elements.
        for i, atom in enumerate(self.atom_params):
            title = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, container_width - 20, element_height),
                                                  text=f"{atom['name'].capitalize()} Settings:",
                                                  manager=self.ui_manager,
                                                  container=self.ui_panel.get_container())
            y_offset += element_height + padding
            
            np_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, 60, element_height),
                                                   text="Count:",
                                                   manager=self.ui_manager,
                                                   container=self.ui_panel.get_container())
            np_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(80, y_offset, container_width - 90, element_height),
                                                           manager=self.ui_manager,
                                                           container=self.ui_panel.get_container())
            np_entry.set_text(str(atom["num_particles"]))
            y_offset += element_height + padding
            
            r_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, 60, element_height),
                                                  text="Radius:",
                                                  manager=self.ui_manager,
                                                  container=self.ui_panel.get_container())
            r_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(80, y_offset, container_width - 90, element_height),
                                                          manager=self.ui_manager,
                                                          container=self.ui_panel.get_container())
            r_entry.set_text(str(atom["radius"]))
            y_offset += element_height + padding
            
            self.ui_elements[i] = {"num_particles": np_entry, "radius": r_entry, "forces": {}}
        
        # Add force matrix controls.
        force_title = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, container_width - 20, element_height),
                                                  text="Force (row->col):",
                                                  manager=self.ui_manager,
                                                  container=self.ui_panel.get_container())
        y_offset += element_height + padding
        
        for i in range(NUM_TYPES):
            row_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, container_width - 20, element_height),
                                                    text=f"{self.atom_params[i]['name']} forces:",
                                                    manager=self.ui_manager,
                                                    container=self.ui_panel.get_container())
            y_offset += element_height + padding
            for j in range(NUM_TYPES):
                f_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(10, y_offset, 80, element_height),
                                                      text=f"to {self.atom_params[j]['name']}:",
                                                      manager=self.ui_manager,
                                                      container=self.ui_panel.get_container())
                f_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect(100, y_offset, container_width - 110, element_height),
                                                              manager=self.ui_manager,
                                                              container=self.ui_panel.get_container())
                f_entry.set_text(str(self.force_params[i][j]))
                self.ui_elements[i]["forces"][j] = f_entry
                y_offset += element_height + padding
            y_offset += padding

        self.update_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(10, y_offset, container_width - 20, 40),
                                                            text="Update Simulation",
                                                            manager=self.ui_manager,
                                                            container=self.ui_panel.get_container())

    def initialize_simulation(self):
        """Rebuild simulation arrays from current atom_params."""
        particle_types = []
        colors = []
        radii = []
        for i, atom in enumerate(self.atom_params):
            count = atom["num_particles"]
            particle_types.extend([i] * count)
            colors.extend([atom["color"]] * count)
            radii.extend([atom["radius"]] * count)
        self.num_particles = len(particle_types)
        self.types = cp.array(particle_types, dtype=cp.int32)
        self.colors = colors
        # Initialize positions within the PLAYGROUND area.
        self.positions = cp.random.uniform(0, PLAYGROUND_WIDTH, (self.num_particles, 2)).astype(cp.float32)
        self.positions[:, 1] = cp.random.uniform(0, Y_SIZE, self.num_particles)
        self.velocities = cp.zeros((self.num_particles, 2), dtype=cp.float32)
        self.radii = cp.array(radii, dtype=cp.float32)

    def update_simulation_from_ui(self):
        """Read UI values and update parameters, then reinitialize simulation."""
        # Get FPS and gravity values
        try:
            new_fps = int(self.ui_elements["fps"].get_text())
            new_gravity = float(self.ui_elements["gravity"].get_text())
            self.fps = new_fps
            
            # Update gravity and recalculate the force matrix
            old_gravity = self.gravity
            self.gravity = new_gravity
            
            # Scale the force matrix to maintain the same relative values but with new gravity
            if old_gravity != 0:  # Avoid division by zero
                scale_factor = new_gravity / old_gravity
                self.force_params = self.force_params * scale_factor
            else:
                # If old gravity was 0, just apply the new gravity directly
                self.force_params = INITIAL_FORCE_MATRIX * new_gravity
                
        except ValueError:
            # If there's an error parsing, keep the current values
            pass

        try:
            self.gravity = float(self.ui_elements["gravity"].get_text())
        except ValueError:
            pass
            
        for i in range(NUM_TYPES):
            try:
                new_count = int(self.ui_elements[i]["num_particles"].get_text())
                new_radius = float(self.ui_elements[i]["radius"].get_text())
            except ValueError:
                continue
            self.atom_params[i]["num_particles"] = new_count
            self.atom_params[i]["radius"] = new_radius

        new_force = np.zeros((NUM_TYPES, NUM_TYPES), dtype=np.float32)
        for i in range(NUM_TYPES):
            for j in range(NUM_TYPES):
                try:
                    new_force[i, j] = float(self.ui_elements[i]["forces"][j].get_text())
                except ValueError:
                    new_force[i, j] = self.force_params[i, j]
        
        self.force_params = new_force
        self.force_matrix = cp.array((self.force_params * self.gravity).flatten())  # Apply gravity here
        self.initialize_simulation()

    def update(self):
        threads_per_block = 256
        blocks_per_grid = (self.num_particles + threads_per_block - 1) // threads_per_block
        
        self.compute_forces_kernel((blocks_per_grid,), (threads_per_block,),
                                   (self.positions, self.velocities, self.types,
                                    self.force_matrix,
                                    np.int32(self.num_particles), np.int32(NUM_TYPES),
                                    np.float32(self.interaction_radius)))
        
        self.positions += self.velocities
        
        self.resolve_collisions_kernel((blocks_per_grid,), (threads_per_block,),
                                       (self.positions, self.velocities,
                                        np.int32(self.num_particles), self.radii))
        
        bounds = cp.array([PLAYGROUND_WIDTH, Y_SIZE], dtype=cp.float32)
        self.positions = cp.clip(self.positions, 0, bounds)
        mask = (self.positions <= 0) | (self.positions >= bounds)
        self.velocities[mask] *= -1

    def draw(self):
        # Draw simulation in the PLAYGROUND area.
        for pos, t in zip(cp.asnumpy(self.positions), cp.asnumpy(self.types)):
            r = self.atom_params[t]["radius"]
            pygame.draw.circle(self.window, self.atom_params[t]["color"], pos.astype(int), int(r))

    def run(self):
        running = True
        simulation_steps = 0
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.ui_manager.process_events(event)
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.update_button:
                        self.update_simulation_from_ui()
            
            time_delta = clock.tick(self.fps) / 1000.0
            self.ui_manager.update(time_delta)
            self.update()
            simulation_steps += 1
            
            # Draw everything only once per frame.
            self.window.fill((0, 0, 0))
            self.draw()
            self.ui_manager.draw_ui(self.window)
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    simulation = GPUParticleSimulation()
    simulation.run()