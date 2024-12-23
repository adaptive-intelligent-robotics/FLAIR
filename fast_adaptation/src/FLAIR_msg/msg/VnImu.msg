builtin_interfaces/Time stamp
float64[3] angular_velocity_dps             # x,y,z
float64[3] linear_acceleration_mps2         # x,y,z
float64[3] roll_pitch_yaw_deg               # x,y,z
float64[3] position_met                     # x,y,z
float64[3] linear_velocity_mps              # x,y,z

# confidence of data on [0,1], when available, otherwise <0 if not
float32 confidence_ori
float32 confidence_acc
float32 confidence_gyro
float32 confidence_pos
float32 confidence_vel
