import tkinter as tk

UNIT = 50

def callback(event):
    print('click at %d, %d' % (event.x, event.y))
    click_x = (event.x // UNIT)*UNIT
    click_y = (event.y // UNIT)*UNIT
    print('the coordinates of the rect is start %d, %d, %d, %d' % (click_x, click_y, click_x+UNIT, click_y+UNIT))
    canvas.create_rectangle(click_x, click_y, click_x+UNIT, click_y+UNIT, fill='red', outline='red')


window = tk.Tk()
window.title = 'the test click'
#window.geometry(100, 100 ,300 ,300)
window.geometry("{}x{}+{}+{}".format(350, 350, 100, 100))
canvas = tk.Canvas(window, bg='white', height=350, width=350)
canvas.bind("<Button-1>", callback)



canvas.pack()
window.mainloop()