classdef Angle_vec
    properties
        angle;
        direct;
    end
    
    methods
        function obj = Angle_vec()
            %empty
        end
    end
    
    methods        
        function d = wise(rot)
           if sign(rot.angle)>0
               d = 'r';
           else
               d = 'l';
           end
        end
    end
end

