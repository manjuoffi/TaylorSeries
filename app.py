from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import sympy as sp

app = Flask(__name__)

def taylor_series(fn, c, n, x):
    x = sp.Symbol('x')
    series = sp.series(fn, x, c, n+1).removeO()
    explanation = ""
    for k in range(n+1):
        term = sp.diff(fn, x, k).subs(x, c) * (x - c)**k / sp.factorial(k)
        explanation += f"Term {k}: ({sp.diff(fn, x, k).subs(x, c)}) * (x - {c})^{k} / {sp.factorial(k)}\n"
    return series, explanation

def plot_taylor_series(fn_str, c, n, x_range):
    x = sp.Symbol('x')
    fn = sp.sympify(fn_str)

    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, 400)

    y_original = [float(sp.N(fn.subs(x, xi))) for xi in x_values]
    series, explanation = taylor_series(fn, c, n, x)
    y_taylor = [float(sp.N(series.subs(x, xi))) for xi in x_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_original, label='Original Function', color='blue')
    plt.plot(x_values, y_taylor, label=f'Taylor Series (Order {n})', color='red', linestyle='--')
    plt.title('Taylor Series Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return img_base64, explanation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fn_input = request.form['function']
        c_input = float(request.form['center'])
        n_input = int(request.form['order'])
        x_min = float(request.form['xmin'])
        x_max = float(request.form['xmax'])

        img_base64, explanation = plot_taylor_series(fn_input, c_input, n_input, (x_min, x_max))
        return render_template('result.html', plot_url=f"data:image/png;base64,{img_base64}", explanation=explanation)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
