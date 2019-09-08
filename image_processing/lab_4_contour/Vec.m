classdef Vec
    properties
        x1;
        y1;
        x2;
        y2;
        len;
    end
    
    methods
        function obj = Vec()
            % empty
        end
    end
    
    methods
        function l = len_vect(obj)
            x = (obj.x1 - obj.x2)^2;
            y = (obj.y1 - obj.y2)^2;
            l = sqrt(x+y);
        end
    end
    
    methods
        function y = rotation_angle(v1,v2)
            x1 = v1.x2 - v1.x1;
            y1 = v1.y2 - v1.y1;
            x2 = v2.x2 - v2.x1;
            y2 = v2.y2 - v2.y1;
            y = atan2(y1,x1)/pi*180-atan2(y2,x2)/pi*180;
            if y > 180 
                y = y-360;
            elseif y<-180
                y = 360+y
            end
%             hold on;
%             plot([0 x1], [0 y1]);
%             hold on;
%             plot([0 x2], [0 y2]);
        end
    end
end
        


