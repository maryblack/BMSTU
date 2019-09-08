% Сохраняем в таблицу индексы реальной точки (real), 
% индекс автоматически выделенной точки (output) и расстояние в пикселях (dist)

hand = imread('my_boundaries.png');
auto = imread('matlab_boundaries.png');
[row, col, color] = size(hand);
dist = zeros(row,0);
real = zeros(row,0);
output = zeros(row,0);
for i=1:row
    s = 0;
    j = 1;
    p_hand = 0;
    p_auto = 0;
    while (p_hand<1)&(j<col)
        if hand(i,j,2)==255
             p_hand = j;
        end
        j = j+1;
    end
    real(i,1) = i;
    real(i,2) = p_hand;
    j=1;
    while (p_auto<1)&(j<col)
        if auto(i,j,1)==255
             p_auto = j;
        end
        j = j+1;
    end
    s = abs(p_hand-p_auto);
   
    output(i,1) = i;
    output(i,2) = p_auto;
    dist(i,1) = s; 
end

T = table(real, output, dist);
low_value = 0;
high_value = 5;
percent = derivation_percent(low_value,high_value,T.dist);
disp('percent');
disp(percent);
function y = derivation_percent(low,high,dist)
    [m,n]=size(dist);
    i = 1;
    rh = 0;
    while i<m
        if dist(i,1)>high
            rh = rh +1;
        end
        i = i+1;
    end
    
    rl = 0;
    i = 1;
    while i<m
        if dist(i,1)>=low
            rl = rl +1;
        end
        i = i+1;
    end
    y = 100*(rl - rh)/m;
end
      
      
