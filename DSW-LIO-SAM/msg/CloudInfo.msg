# Cloud Info
std_msgs/Header header 

int32[] start_ring_index
int32[] end_ring_index

int32[]  point_col_ind # point column index in range image
float32[] point_range # point range 

int64 imu_available
int64 odom_available

# Attitude for LOAM initialization
float32 imu_roll_init
float32 imu_pitch_init
float32 imu_yaw_init

# Initial guess from imu pre-integration
float32 initial_guess_x
float32 initial_guess_y
float32 initial_guess_z
float32 initial_guess_roll
float32 initial_guess_pitch
float32 initial_guess_yaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

# ===== DSW-LIO-SAM: 语义字段 =====
uint32[] point_semantic_labels    # 每个点的语义标签（SemanticLabel枚举值）
float32[] point_semantic_weights  # 每个点的语义权重 w_i
# ===== DSW-LIO-SAM 结束 =====