import tkinter as tk
from PIL import Image, ImageTk
from sys import argv

window = tk.Tk(className="bla")

img_dir = "../../poses/labelled/neutral_5.png"
img = Image.open(img_dir)
w,h = img.size
img = img.resize((w, h),Image.ANTIALIAS)
canvas = tk.Canvas(window, width=img.size[0], height=img.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(img)
w_,h_ = img.size
canvas.create_image(w_//2, h_//2, image=image_tk)

points = list()
ops = list()
i=0
def callback(event):
    print("clicked at: ", event.x, event.y)
    #draw a point
    x1, y1 = (event.x - 3), (event.y - 3)
    x2, y2 = (event.x + 3), (event.y + 3)
    id = canvas.create_oval(x1, y1, x2, y2, fill="red")
    ops.append(id)
    #create the text
    global i
    x3, y3 = (event.x - 9), (event.y - 9)
    id = canvas.create_text(x3,y3,text=str(i),fill="cyan",font="Times 10 italic bold")
    ops.append(id)
    i += 1
    points.append((event.x,event.y))

def undo(event):
    global i
    del points[i-1]
    canvas.delete(ops[2*i-1])
    canvas.delete(ops[2*i-2])
    del ops[2*i-1]
    del ops[2*i-2]
    i -= 1

canvas.bind("<Button-1>", callback)
canvas.bind("<Button-3>", undo)
tk.mainloop()

output_dir = img_dir[:-4]+".pts"
with open(output_dir,'w') as f:
    for p in points:
        f.write(str(p[0])+','+str(p[1])+'\n')

