function [y0, y1, y2, y3] = simpQamSlicer256QAM(x, h)
    y0 = x;
    y1 = -abs(x) + 8*h;
    y2 = -abs(abs(x)-8*h)+4*h;  
    y3 = -abs(abs(abs(x)-8*h)-4*h)+2*h;  
end