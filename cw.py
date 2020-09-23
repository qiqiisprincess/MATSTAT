import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 80
X = np.arange(-3.9, 4 + 1e-10, 0.1)
REPS = 1000
A = 2
B = 2


def lad(smp_x, smp_y):
    def med(smp):
        return (smp[N // 2 - 1] + smp[N // 2]) / 2 if N % 2 == 0 else smp[(N - 1) // 2]

    def sgn(num):
        return 1 if num > 0 else (0 if num == 0 else -1)

    med_x, med_y = med(sorted(smp_x)), med(sorted(smp_y))
    sgn_x = [sgn(i - med_x) for i in smp_x]
    sgn_y = [sgn(i - med_y) for i in smp_y]
    r_Q = sum(np.multiply(sgn_x, sgn_y)) / N
    _l = N // 4 if N % 4 == 0 else N // 4 + 1
    j = N - _l + 1
    qy_by_qx = (smp_y[j] - smp_y[_l]) / (smp_x[j] - smp_x[_l])
    b = r_Q * qy_by_qx
    a = med_y - b * med_x
    return {'a': a, 'b': b}


def get_y_with_outliers(num_of_outliers):
    te = ss.norm.rvs(size=N, loc=0, scale=1)
    ty = A + B * X + te
    i_o = np.random.choice(N, num_of_outliers) if num_of_outliers > 0 else []
    for i in i_o:
        ty[i] += 10 if ty[i] >= 0 else -10
    return ty


def collect_info(num_of_outliers, func=lad):
    a, b = [], []
    for _ in range(REPS):
        y = get_y_with_outliers(num_of_outliers)
        res = func(X, y)
        a.append(res['a'])
        b.append(res['b'])

    av_a = sum(a) / REPS
    av_b = sum(b) / REPS
    cola = [np.abs(A - _a) for _a in a]
    ea = sum(cola) / REPS
    colb = [np.abs(B - _b) for _b in b]
    eb = sum(colb) / REPS

    cola = [_a ** 2 for _a in cola]
    colb = [_b ** 2 for _b in colb]
    ea2 = sum(cola) / REPS
    eb2 = sum(colb) / REPS
    da = ea2 - ea ** 2
    db = eb2 - eb ** 2
    return {'av': [av_a, av_b], 'e': [ea, eb], 'd': [da, db]}


def research():
    a, b = [], []
    eas, ebs = [], []
    das, dbs = [], []
    outliers_in_nums = range(N // 4 + 1)
    n = len(outliers_in_nums)
    for o in outliers_in_nums:
        res = collect_info(o)
        a.append(res['av'][0])
        b.append(res['av'][1])
        eas.append(res['e'][0])
        ebs.append(res['e'][1])
        das.append(res['d'][0])
        dbs.append(res['d'][1])

    outliers_in_percentage = [o / N * 100 for o in outliers_in_nums]
    pd.options.display.float_format = '{:.2f}'.format
    df = pd.DataFrame(columns=['outliers, %'], data=outliers_in_percentage)
    df['average_A'] = np.asarray(a)
    df['average_B'] = np.asarray(b)
    df['E(da)'] = np.asarray(eas)
    df['E(db)'] = np.asarray(ebs)
    df['D(da)'] = np.asarray(das)
    df['D(db)'] = np.asarray(dbs)

    df.to_csv('cw_data.txt', header=True, index=False, sep='\t', float_format='%.2f')

    info_types = ['av', 'e', 'd']
    for info in info_types:
        fig, ax = plt.subplots(1, 1)
        plt.xticks([o for o in outliers_in_nums if o % 4 == 0],
                   ['{:.0f}%'.format(o / N * 100) for o in outliers_in_nums if o % 4 == 0])
        if info == 'av':
            plt.plot(outliers_in_nums, [A for _ in range(N // 4 + 1)], color='#0000FF')
            plt.plot(outliers_in_nums, a, color='#AA4D71')
            plt.plot(outliers_in_nums, b, color='#000000')
            ax.set_title('Среднее значение вычисленных коэффициентов лин. регрессии')
            ax.set_ylabel('Величина коэффициентов')
            ax.legend(['Эталонные a и b', 'Среднее a', 'Среднее b'])
        elif info == 'e':
            plt.plot(outliers_in_nums, eas, color='#AA4D71')
            plt.plot(outliers_in_nums, ebs, color='#000000')
            ax.set_title('Мат. ожидания абсолютных отклонений\nвычисленных коэффициентов от эталонных')
            ax.set_ylabel('Величина мат. ожидания')
            ax.legend([r'$E(\delta_a)$', r'$E(\delta_b)$'])
        else:
            plt.plot(outliers_in_nums, das, color='#AA4D71')
            plt.plot(outliers_in_nums, dbs, color='#000000')
            ax.set_title('Дисперсии абсолютных отклонений\nвычисленных коэффициентов от эталонных')
            ax.set_ylabel('Величина дисперсии')
            ax.legend([r'$D(\delta_a)$', r'$D(\delta_b)$'])

        ax.set_xlabel('Доля выбросов')
        ax.set_facecolor('#d8dcd6')
        ax.grid()
        plt.show()


research()
