% 求取EKF的状态转移方程和状态-测量转换方程的雅可比jacobian矩阵A和H
% 本脚本是基于单车模型运动学的方程建立, 状态向量有如下表述:
% [x y v a theta sigma w l h z] 均在nuscenes实际坐标系下描述
% 其中x是物体的x轴位置
% 其中y是物体的y轴位置
% 其中v是物体的总速度，其值为vx和vy的矢量和
% 其中a是物体的总加速度，其值为ax和ay的矢量和，粗略认为加速度和速度同向
% 其中theta为物体偏航角，即为绕世界坐标系Z轴与正X轴的夹角，逆时针为正
% 其中sigma为物体前轮与车辆中线的转向角


% -----------先求状态转移方程对状态向量的雅可比矩阵A-------------
syms x y v a theta sigma beta omega lr lf T KT w l h z 

% theta0, v0为t = KT(积分下线时的物体偏移角和速度)
syms theta0 v0 t deltax deltay
syms vx vy w_r lf_r l geo2gra_dist

% beta为车辆速度方向与车辆中线的夹角, lr为车辆重心到后轮的距离, lf为车辆重心到前轮的距离
% lr = l * w_r * (0.5 - lf_r);
% lf = l * w_r * lf_r;
% beta = atan( lr / (lr + lf) * tan(sigma));
v = v0 + a * (t - KT + T);
theta = theta0 + v0 / lr * sin(beta) * (t - KT + T);

% 求deltax和deltay
ft_x = v0 * cos(beta + theta);
deltax = int(ft_x, t, KT-T, KT);
ft_y = v0 * sin(beta + theta);
deltay = int(ft_y, t, KT-T, KT);
ft_the = v / lr * sin(beta);
deltathe = int(ft_the, t, KT-T, KT);


% 定义系统状态向量state vector
s_v = [x y v0 a theta0 sigma w l h z];
% 定义状态转移方程f_t
f_t = [x + deltax; 
       y + deltay; 
       v0 + a * T; 
       a; 
       theta0 + deltathe;
       sigma;
       w; l; h; z];
% 获取雅可比矩阵A
A = jacobian(f_t, s_v)
x + deltax
y + deltay
theta0 + deltathe
% ---------------------------------------------------------

% -----------先求状态转移方程对状态向量的雅可比矩阵A（sigma和v为极小值的情况）-------------
% 定义系统状态向量state vector
% 定义状态转移方程f_t
f_t_zero = [x + (v0 * T + a * T^2 / 2) * cos(theta0); 
            y + (v0 * T + a * T^2 / 2) * sin(theta0); 
            v0 + a * T;
            a;
            theta0; 
            sigma; w; l; h; z];
% 获取雅可比矩阵A
A_zero = jacobian(f_t_zero, s_v)
% ---------------------------------------------------------



% ---------再求状态->测量转换方程对状态向量的雅可比矩阵H----------
% 测量向量有如下表述: [x, y, vx, vy, theta]
geo2gra_dist = l * w_r * (0.5 - lf_r)
% 定义状态->测量转换方程h_t
h_t = [x - geo2gra_dist * cos(theta0); 
       y - geo2gra_dist * sin(theta0); 
       v0 * cos(theta0 + beta); 
       v0 * sin(theta0 + beta); 
       theta0; w; l; h; z];
% 获取雅可比矩阵H
H = jacobian(h_t, s_v)
% ---------------------------------------------------------
% 测量向量有如下表述: [x, y, theta]
% 定义状态->测量转换方程h_t
h_t_novel = [x - geo2gra_dist * cos(theta0); y - geo2gra_dist * sin(theta0); theta0; w; l; h; z];
% 获取雅可比矩阵H
H_novel = jacobian(h_t_novel, s_v)