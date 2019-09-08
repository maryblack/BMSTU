HEART = 'heart.png';

image = imread(HEART);
image = rgb2gray(image);
[row,col] = size(image);
N = row*col;
contur_points = zeros(N,0);
contur_map = containers.Map('KeyType','double', 'ValueType','any');
key_converter = @(h,l) sub2ind([300,300], h,l);
start = zeros(2,0);
i = 1;
while i<row
    j =1;
    while j<col
        if image(i,j)==255
            start = [i,j];
            i = row;
            j = col;
        end
        j = j+1;
    end
    i = i+1;
end
x_start = start(1,1);
y_start = start(1,2);
 
adj_point = zeros(8,0);
curr = start;
next = [0,0];
k = 1;
j = 1;
while 1
    i = curr(1,1);
    j = curr(1,2);
    contur_points(k,1)=i;
    contur_points(k,2)=j;
    contur_map(key_converter(i,j)) = 1;
    M = image(i-1:i+1,j-1:j+1);
    adj_point = explication(M);
 
    n = 1;
    while n<8
        [curr_point_i, cuur_point_j] = recieve_point(n,i,j);
        not_visited = isKey(contur_map, key_converter(curr_point_i, cuur_point_j));
        if adj_point(n)==255 & not_visited == 0
           break;
        end
        n = n+1;
    end
    [next_i,next_j] = recieve_point(n,i,j);
    next(1) = next_i;
    next(2) = next_j;
    curr = next;
    
    if isKey(contur_map, key_converter(next_i,next_j))== 1
        break;
    end
    k = k +1;
    
end
%disp('done');
 
[rows, cols] = size(contur_points(1:k,1:2));
xs = zeros(rows, 0);
ys = zeros(rows, 0);
i = 1;
while i < rows
    xs(i) = contur_points(i, 2); 
    ys(i) = row - contur_points(i, 1);
    i = i + 1;
end
% plot(xs, ys, '.');
 
STEP = 14;
[vectors vector_number] = vectorize(xs, ys, STEP, k); 
 
i = 1;
hold on;
while i <= (vector_number+1)
    v = vectors(i);
    %disp(sprintf("[%d:%d] --> [%d:%d]", v.x1, v.y1, v.x2, v.y2));
    plot([v.x1 v.x2], [v.y1, v.y2]);
    i = i + 1;
end
i =1;
angles(vector_number, 1) = Angle_vec();
disp(sprintf("\n"));
disp('----Кодирование по трем признакам----');
while i <= (vector_number)
    v = vectors(i);
    if i ==vector_number
        v_next = vectors(1);
    else
        v_next = vectors(i+1);
    end
    rotate = rotation_angle(v,v_next);
    angles(i).angle = rotate;
    angles(i).direct = wise(angles(i));
    disp(sprintf("vect_num:%d --> %f grad, %s, len:%f", i, rotate,angles(i).direct, v.len));
    i = i + 1;
end
 
i = 1;
disp(sprintf("\n"));
disp('----Кодирование координатами концов векторов----');
while i <= (vector_number+1)
    v = vectors(i);
    disp(sprintf("vect_num:%d --> [%d:%d]", i, v.x2, v.y2));
    i = i + 1;
end
i = 1;
 
disp(sprintf("\n"));
disp('----Кодирование полярными координатами----');
angles(vector_number+1, 1) = Angle_vec();
v_0 = Vec();
v_0.x1 = 0;
v_0.y1 = 0;
v_0.x2 = 1;
v_0.y2 = 0;
v_0.len = len_vect(v_0);
while i <= (vector_number+1)
    v = vectors(i);
    x = (xs(1) - v.x2)^2;
    y = (ys(1) - v.y2)^2;
    l = sqrt(x+y);
    rotate = rotation_angle(v_0,v);
    angles(i).angle = rotate;
    angles(i).direct = wise(angles(i));
    dir = angles(i).direct;
    if dir == 'r'
        rotate = 360-abs(rotate);
    elseif dir == 'l'
        rotate = abs(rotate);
    end
    disp(sprintf("vect_num:%d --> angle = %f , r:%f", i, rotate, l));
    i = i + 1;
end
 
i=1;
disp(sprintf("\n"));
disp('----Кодирование трехразрядным кодом----');
num = 0;
while i <= (vector_number+1)
    v = vectors(i);
    x = (xs(1) - v.x2)^2;
    y = (ys(1) - v.y2)^2;
    l = sqrt(x+y);
    rotate = rotation_angle(v_0,v);
    angles(i).angle = rotate;
    angles(i).direct = wise(angles(i));
    dir = angles(i).direct;
    if dir == 'r'
        rotate = 360-abs(rotate);
    elseif dir == 'l'
        rotate = abs(rotate);
    end
    num = freeman_code(rotate);
    disp(sprintf("vect_num:%d --> num = %s ",i,three(num)));
    i = i + 1;
end
 
 
i=1;
disp(sprintf("\n"));
disp('----Кодирование комплексными числами----');
num = 0;
while i <= (vector_number+1)
    v = vectors(i);
    x = (xs(1) - v.x2)^2;
    y = (ys(1) - v.y2)^2;
    l = sqrt(x+y);
    rotate = rotation_angle(v_0,v);
    angles(i).angle = rotate;
    angles(i).direct = wise(angles(i));
    dir = angles(i).direct;
    if dir == 'r'
        rotate = 360-abs(rotate);
    elseif dir == 'l'
        rotate = abs(rotate);
    end
    num = freeman_code(rotate);
    disp(sprintf("vect_num:%d --> num = %s ",i,complex_num(num)));
    i = i + 1;
end
 
i=1;
disp(sprintf("\n"));
disp('----Кодирование проекциями----');
num = 0;
while i <= (vector_number+1)
    v = vectors(i);
    x = (xs(1) - v.x2)^2;
    y = (ys(1) - v.y2)^2;
    l = sqrt(x+y);
    rotate = rotation_angle(v_0,v);
    angles(i).angle = rotate;
    angles(i).direct = wise(angles(i));
    dir = angles(i).direct;
    if dir == 'r'
        rotate = 360-abs(rotate);
    elseif dir == 'l'
        rotate = abs(rotate);
    end
    num = freeman_code(rotate);
    disp(sprintf("vect_num:%d --> num = %s ",i,projection_num(num)));
    i = i + 1;
end
 
 
 
function [vects vect_len] = vectorize(xs, ys, step, len)
    i = 1;
    v_i = 1;
    vect_len = fix(len / step);
    vects(vect_len + 1, 1) = Vec();
    while i < len - step
        x1 = xs(i);
        y1 = ys(i);
        x2 = xs(i + step);
        y2 = ys(i + step);
        vects(v_i).x1 = x1;
        vects(v_i).y1 = y1;
        vects(v_i).x2 = x2;
        vects(v_i).y2 = y2;
        l = len_vect(vects(v_i));
        vects(v_i).len = l;
        v_i = v_i + 1;
        i = i + step;
    end
    
    latest = vects(v_i - 1);
%     manually adjusting last connection
    vects(v_i).x1 = latest.x2;
    vects(v_i).y1 = latest.y2;
    vects(v_i).x2 = vects(1).x1;
    vects(v_i).y2 = vects(1).y1;
    l = len_vect(vects(v_i));
    vects(v_i).len = l;
    %disp('asda');
end
        
 
function E = explication(M)
    E = zeros(8,0);
    E(1) = M(2,3);
    E(2) = M(3,3);
    E(3) = M(3,2);
    E(4) = M(3,1);
    E(5) = M(2,1);
    E(6) = M(1,1);
    E(7) = M(1,2);
    E(8) = M(1,3);
end
 
function [next_i, next_j] = recieve_point(num,i,j)
    if num == 1
        next_i = i;
        next_j = j+1;
    
    elseif num == 2
        next_i = i+1;
        next_j = j+1;
    
    elseif num == 3
        next_i = i+1;
        next_j = j;
    
    elseif num == 4
        next_i = i+1;
        next_j = j-1;
    
    elseif num == 5
        next_i = i;
        next_j = j-1;
   
    elseif num == 6
        next_i = i-1;
        next_j = j-1;
   
    elseif num == 7
        next_i = i-1;
        next_j = j;
    
    elseif num == 8
        next_i = i-1;
        next_j = j+1;
    else 
        next_i = i;
        next_j = j;  
    end
  
end
function num = freeman_code(angle)
    if (angle >= 0 && angle < 22.5) || (angle >= 337.5 && angle <= 360)
        num = 0;
    elseif angle >= 22.5 && angle < 67.5
        num = 7;
    elseif angle >= 67.5 && angle < 112.5
        num = 6;
    elseif angle >= 112.5 && angle < 157.5
        num = 5;
    elseif angle >= 157.5 && angle < 202.5
        num = 4;
    elseif angle >= 202.5 && angle < 247.5
        num = 3;
    elseif angle >= 247.5 && angle < 292.5
        num = 2;
    elseif angle >= 292.5 && angle < 337.5
        num = 1;         
    end
end
function ch = three(num);
    if num == 0
        ch = '000';   
    elseif num == 1
        ch = '001';  
    elseif num == 2
        ch = '010';
    elseif num == 3
        ch = '011';    
    elseif num == 4
        ch = '100';
    elseif num == 5
        ch = '101';
    elseif num == 6
        ch = '110';   
    elseif num == 7
        ch = '111'; 
    else
        ch = 'error_num';
    end
end
 
function ch = complex_num(num);
    if num == 0
        ch = '0+0i';   
    elseif num == 1
        ch = '1-1i';  
    elseif num == 2
        ch = '0-1i';
    elseif num == 3
        ch = '-1-1i';    
    elseif num == 4
        ch = '-1+0i';
    elseif num == 5
        ch = '-1+1i';
    elseif num == 6
        ch = '0+1i';   
    elseif num == 7
        ch = '1+1i'; 
    else
        ch = 'error_num';
    end
end
 
function ch = projection_num(num);
    if num == 0
        ch = '(0,0)';   
    elseif num == 1
        ch = '(1,-1)';  
    elseif num == 2
        ch = '(0,-1)';
    elseif num == 3
        ch = '(-1,-1)';    
    elseif num == 4
        ch = '(-1,0)';
    elseif num == 5
        ch = '(-1,1)';
    elseif num == 6
        ch = '(0,1)';   
    elseif num == 7
        ch = '(1,1)'; 
    else
        ch = 'error_num';
    end
end
 
