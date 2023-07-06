% 求取EKF的状态转移方程和状态-测量转换方程的雅可比jacobian矩阵A和H
% 本脚本是基于CTRA模型运动学的方程建立, 状态向量有如下表述:
% [x, y, theta, v, a, omega] 均在nuscenes实际坐标系下描述
% 其中x是物体的x轴位置
% 其中y是物体的y轴位置
% 其中v是物体的总速度，其值为vx和vy的矢量和
% 其中a是物体的总加速度，其值为ax和ay的矢量和，粗略认为加速度和速度同向
% 其中theta为物体偏航角，即为绕世界坐标系Z轴与正X轴的夹角，逆时针为正
% 其中omega是角加速度


% -----------先求状态转移方程对状态向量的雅可比矩阵A-------------
syms x y v a theta omega deltat w h l z

% 定义系统状态向量state vector
s_v = [x y theta v a omega w l h z];
% 定义状态转移方程f_t
f_t = [x + 1 / omega^2 * ( (v * omega + a * omega * deltat) * sin(theta + omega * deltat) + a * cos(theta + omega * deltat) - v * omega * sin(theta) - a * cos(theta) ); 
       y + 1 / omega^2 * ( (-v * omega - a * omega * deltat) * cos(theta + omega * deltat) + a * sin(theta + omega * deltat) + v * omega * cos(theta) - a * sin(theta) ); 
       theta + omega * deltat; 
       v + a * deltat; 
       a;
       omega; 
       w; l; h; z];
% 获取雅可比矩阵A
A = jacobian(f_t, s_v)
% 获取雅可比矩阵A的转置A_T
A_T = A.';
% ---------------------------------------------------------




% -----------先求状态转移方程对状态向量的雅可比矩阵A（omega为极小值的情况）-------------
% 定义系统状态向量state vector
s_v = [x y theta v a omega w l h z];
% 定义状态转移方程f_t
f_t_zero = [x + (v * deltat + a * deltat^2 / 2) * cos(theta); 
            y + (v * deltat + a * deltat^2 / 2) * sin(theta); 
            theta + omega * deltat; 
            v + a * deltat; 
            a;
            omega;
            w; l; h; z];
% 获取雅可比矩阵A
A_zero = jacobian(f_t_zero, s_v)
% 获取雅可比矩阵A的转置A_T
A_zero_T = A_zero.';
% ---------------------------------------------------------


% ---------再求状态->测量转换方程对状态向量的雅可比矩阵H----------
% 测量向量有如下表述: [x, y, vx, vy, theta]
syms vx vy

% 定义状态->测量转换方程h_t
h_t = [x; 
       y; 
       v * cos(theta); 
       v * sin(theta); 
       theta;
       w; l; h; z];
% 获取雅可比矩阵H
H = jacobian(h_t, s_v)
% ---------------------------------------------------------
% 定义状态->测量转换方程h_t
h_t_novel = [x; 
       y; 
       theta;
       w; l; h; z];
% 获取雅可比矩阵H
H_novel = jacobian(h_t_novel, s_v)