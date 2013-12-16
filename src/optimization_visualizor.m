function varargout = optimization_visualizor(varargin)
% OPTIMIZATION_VISUALIZOR MATLAB code for optimization_visualizor.fig
%      OPTIMIZATION_VISUALIZOR, by itself, creates a new OPTIMIZATION_VISUALIZOR or raises the existing
%      singleton*.
%
%      H = OPTIMIZATION_VISUALIZOR returns the handle to a new OPTIMIZATION_VISUALIZOR or the handle to
%      the existing singleton*.
%
%      OPTIMIZATION_VISUALIZOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in OPTIMIZATION_VISUALIZOR.M with the given input arguments.
%
%      OPTIMIZATION_VISUALIZOR('Property','Value',...) creates a new OPTIMIZATION_VISUALIZOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before optimization_visualizor_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to optimization_visualizor_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help optimization_visualizor

% Last Modified by GUIDE v2.5 13-Oct-2013 17:05:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @optimization_visualizor_OpeningFcn, ...
                   'gui_OutputFcn',  @optimization_visualizor_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before optimization_visualizor is made visible.
function optimization_visualizor_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to optimization_visualizor (see VARARGIN)

% Choose default command line output for optimization_visualizor
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes optimization_visualizor wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = optimization_visualizor_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in read_data.
function read_data_Callback(hObject, eventdata, handles)
% hObject    handle to read_data (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, path] = uigetfile('*.*', '读取文件', '../data');
if path == 0
    return;
end
set(handles.cordinate, 'UserData', dlmread([path, filename]));



function interval_Callback(hObject, eventdata, handles)
% hObject    handle to interval (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of interval as text
%        str2double(get(hObject,'String')) returns contents of interval as a double


% --- Executes during object creation, after setting all properties.
function interval_CreateFcn(hObject, eventdata, handles)
% hObject    handle to interval (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in show_param_w.
function show_param_w_Callback(hObject, eventdata, handles)
% hObject    handle to show_param_w (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
objs = findobj(handles.cordinate, 'Type', 'line');
if ~isempty(objs)
    delete(objs);
end
data = get(handles.cordinate, 'UserData');
intv = str2num(get(handles.interval, 'String'));
set(handles.show_status, 'String', '正在绘图...');
for i = 2:size(data, 1)
    line('XData', data(i-1:i, 1), 'YData', data(i-1:i, 2), ...
        'Marker', '.', 'MarkerSize', 6);
    pause(intv);
end
set(handles.show_status, 'String', '完成');


% --- Executes on mouse motion over figure - except title and menu.
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cp = get(handles.cordinate, 'CurrentPoint');
x = cp(1, 1); y = cp(1, 2);

str = sprintf('坐标：[%.3f, %.3f]', x, y);
set(handles.show_cordinate, 'String', str);


% --- Executes on button press in show_param_b.
function show_param_b_Callback(hObject, eventdata, handles)
% hObject    handle to show_param_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
objs = findobj(handles.cordinate, 'Type', 'line');
if ~isempty(objs)
    delete(objs);
end
data = get(handles.cordinate, 'UserData');
intv = str2num(get(handles.interval, 'String'));
set(handles.show_status, 'String', '正在绘图...');
for i = 2:size(data, 1)
    line('XData', i-1:i, 'YData', data(i-1:i, 3), ...
        'Marker', '.', 'MarkerSize', 6);
    pause(intv);
end
set(handles.show_status, 'String', '完成');


% --- Executes on button press in toggle_zoom.
function toggle_zoom_Callback(hObject, eventdata, handles)
% hObject    handle to toggle_zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject, 'Value')
    if get(handles.toggle_pan, 'Value')
        pan off;
        set(handles.toggle_pan, 'Value', 0);
    end
    zoom on;
else
    zoom off;
end
% Hint: get(hObject,'Value') returns toggle state of toggle_zoom


% --- Executes on button press in toggle_pan.
function toggle_pan_Callback(hObject, eventdata, handles)
% hObject    handle to toggle_pan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(hObject, 'Value')
    if get(handles.toggle_zoom, 'Value')
        zoom off;
        set(handles.toggle_zoom, 'Value', 0);
    end
    pan on;
else
    pan off;
end
% Hint: get(hObject,'Value') returns toggle state of toggle_pan


% --- Executes on button press in show_obj.
function show_obj_Callback(hObject, eventdata, handles)
% hObject    handle to show_obj (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
objs = findobj(handles.cordinate, 'Type', 'line');
if ~isempty(objs)
    delete(objs);
end
data = get(handles.cordinate, 'UserData');
intv = str2num(get(handles.interval, 'String'));
set(handles.show_status, 'String', '正在绘图...');
for i = 2:size(data, 1)
    line('XData', i-1:i, 'YData', data(i-1:i, 4), ...
        'Marker', '.', 'MarkerSize', 6);
    pause(intv);
end
set(handles.show_status, 'String', '完成');


% --- Executes during object creation, after setting all properties.
function cordinate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cordinate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate cordinate


% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
