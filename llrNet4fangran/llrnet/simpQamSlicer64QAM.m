function [y0, y1, y2] = simpQamSlicer64QAM(x, h)
    y0 = x;
    y1 = -abs(x) + 4*h;
    y2 = -abs(abs(x)-4*h)+2*h;  
end