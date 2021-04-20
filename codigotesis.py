import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
from scipy.optimize import differential_evolution
from scipy.spatial.distance import euclidean


def mae(predi, val):
    return np.sum(np.abs(predi - val)) / len(predi)


def primnero(vecti):
    return vecti[0]


def createDB(seri, w, tau=1, delta=2):
    N = len(seri)
    movi = delta - 1
    resul = np.zeros((N-((w-1)*tau+1)-movi, w+1))
    for i in range(N-((w-1)*tau+1)-movi):
        resul[i] = np.concatenate((seri[i: i+((w-1)*tau+1): tau], np.array([seri[i+((w-1)*tau+1)+movi]])))
    return resul


def liftPredict(miniser):
    res = np.copy(miniser)
    res[: -1] = miniser[: -1] + miniser[1:]
    return res


def liftUpdate(miniser):
    res = np.copy(miniser)
    res[1:] = miniser[1:] + miniser[: -1]
    return res


def wLift(seriet, m, coeffsarr):
    dl = seriet[0::2]
    sl = seriet[1::2]
    if m*2+1 != len(coeffsarr):
        print("La regaste con las dimensiones")
        return 1
    else:
        for i in range(int(len(coeffsarr)/2)):
            dl = dl - coeffsarr[2*i] * liftPredict(sl)
            sl = sl + coeffsarr[2*i+1] * liftUpdate(dl)
        sl = sl*coeffsarr[-1]
        dl = dl/coeffsarr[-1]
        return sl, dl


def wLiftShift(seriet, m, coeffsarr):
    sl = seriet[0::2]
    dl = seriet[1::2]
    if m*2+1 != len(coeffsarr):
        print("La regaste con las dimensiones")
        return 1
    else:
        for i in range(int(len(coeffsarr)/2)):
            dl = dl - coeffsarr[2*i] * liftPredict(sl)
            sl = sl + coeffsarr[2*i+1] * liftUpdate(dl)
        sl = sl*coeffsarr[-1]
        dl = dl/coeffsarr[-1]
        return sl, dl


def wLiftInv(aprox, deta, m, coeffsarr):
    dl = coeffsarr[-1]*deta
    sl = aprox/coeffsarr[-1]
    for i in reversed(range(int(len(coeffsarr)/2))):
        sl = sl - coeffsarr[2*i+1] * liftUpdate(dl)
        dl = dl + coeffsarr[2*i] * liftPredict(sl)
    serrriori = np.zeros(2*len(dl))
    serrriori[0::2] = dl
    serrriori[1::2] = sl
    return serrriori


def wLiftInvShift(aprox, deta, m, coeffsarr):
    dl = coeffsarr[-1]*deta
    sl = aprox/coeffsarr[-1]
    for i in reversed(range(int(len(coeffsarr)/2))):
        sl = sl - coeffsarr[2*i+1] * liftUpdate(dl)
        dl = dl + coeffsarr[2*i] * liftPredict(sl)
    serrriori = np.zeros(2*len(dl))
    serrriori[1::2] = dl
    serrriori[0::2] = sl
    return serrriori


def wLiftRed(seriet, m, coeffsarr):
    if len(seriet) % 2 != 0:
        print("Nomás no.")
        return 1
    aproe, detae = wLift(seriet, m, coeffsarr)
    aproo, detao = wLiftShift(seriet, m, coeffsarr)
    apro = np.zeros(len(seriet))
    deta = np.zeros(len(seriet))
    apro[0::2] = aproo
    apro[1::2] = aproe
    deta[0::2] = detao
    deta[1::2] = detae
    return apro, deta


def wLiftRedInv(aprox, deta, m, coeffsarr):
    aproo = aprox[0::2]
    detao = deta[0::2]
    aproe = aprox[1::2]
    detae = deta[1::2]
    seriete1 = wLiftInv(aproe, detae, m, coeffsarr)
    seriete2 = wLiftInvShift(aproo, detao, m, coeffsarr)
    if not np.array_equal(seriete1, seriete2):
        seriete = (seriete1 + seriete2) / 2
    else:
        seriete = seriete1
    return seriete


def wLiftn(seriet, m, coeffsarr, lvl):
    misierirs = []
    targserie = np.copy(seriet)
    for i in range(lvl):
        apro, deta = wLift(targserie, m, coeffsarr)
        misierirs.append(deta)
        targserie = apro
    misierirs.append(apro)
    return misierirs


def wLiftnRed(seriet, m, coeffsarr, lvl):  # cAn, cDn, ..., cD1
    minisers = []
    targserie = np.copy(seriet)
    for i in range(lvl):
        apro, deta = wLiftRed(targserie, m, coeffsarr)
        minisers.insert(0, deta)
        targserie = np.copy(apro)
    minisers.insert(0, apro)
    return minisers


def wLiftnInv(miniers, m, coeffsarr):
    aprox = miniers[-1]
    for i in reversed(range(len(miniers)-1)):
        aprox = wLiftInv(aprox, miniers[i], m, coeffsarr)
    return aprox


def wLiftnRedInv(miniers, m, coeffsarr):
    aprox = miniers[0]
    for i in range(1, len(miniers)):
        aprox = wLiftRedInv(aprox, miniers[i], m, coeffsarr)
    return aprox


def gedBffs(retarded, beton, bffsno):
    procesinos = [[euclidean(retarded, beton[i][:-1]), beton[i][-1]] for i in range(len(beton))]
    procesinos.sort(key=primnero)
    return np.mean(np.array(procesinos)[:, 1][:bffsno])


def gedBefos(retarded, beton, bffsno, numpro):
    betono = len(beton)
    procesinos = []
    regulator = 96 - numpro - (betono % 96)
    for i in range(betono):
        if ((i + 5 + regulator) % 96) <= 10:
            procesinos.append([euclidean(retarded, beton[i][:-1]), beton[i][-1]])
    procesinos.sort(key=primnero)
    return np.mean(np.array(procesinos)[:, 1][:bffsno])


def beffregression(superretarded, beton, bffsno, modi=True):
    teacher = len(superretarded)
    cuatromolde = np.zeros(teacher)
    if modi:
        for i in range(teacher):
            cuatromolde[i] = gedBefos(superretarded[i], beton, bffsno, i)
    else:
        for i in range(teacher):
            cuatromolde[i] = gedBffs(superretarded[i], beton, bffsno)
    return cuatromolde


def vecilift(kip, eme, numerodeamigos, extra):
    evo = extra
    np.random.seed(70)
    warnings.simplefilter('ignore')
    np.set_printoptions(suppress=True)
    file_path = '../series/DEMANDA_NETA_SIN.csv'
    df_all = pd.read_csv(file_path, index_col='FECHA')
    idx_sub_sampling = [i for i in range(0, df_all.__len__(), 15)]
    valsdec = np.copy(kip).tolist()
    lsteps = int(len(valsdec)/2)
    valsdec.insert(0, lsteps)
    wlevel = 3
    errpars = []
    bunbunblack = 92
    ser_val = np.zeros(bunbunblack)
    pred = np.zeros(bunbunblack)
    for name_time_series in ['DEMANDA_CEL', 'DEMANDA_NES', 'DEMANDA_NOR', 'DEMANDA_NTE', 'DEMANDA_OCC', 'DEMANDA_ORI', 'DEMANDA_PEN']:
        df = df_all[name_time_series][idx_sub_sampling]
        df = df[:df.index.get_loc('2018-06-25 00:00:00')]
        time_series_scaler = MinMaxScaler()
        scaled_time_series_values = time_series_scaler.fit_transform(df.values.reshape(-1, 1))
        scaled_time_series = pd.Series(scaled_time_series_values.ravel(), index=df.index)
        time_series = scaled_time_series.values
        maxts = np.amax(time_series)
        forcastini = np.zeros(bunbunblack)
        m = eme
        bajito = []
        for i in range(bunbunblack):
            bajito.append(wLiftnRed(np.copy(time_series[i: -bunbunblack-3-1+i]), valsdec[0], valsdec[1:], wlevel)[0])
        for k in range(len(bajito)):
            betillo = createDB(bajito[k], m)
            betval = [bajito[k][-m:]]
            busnils = beffregression(betval, betillo, numerodeamigos, True)
            bajito[k] = np.append(bajito[k], busnils)
        for i in range(len(bajito)):
            maxbaj = np.amax(bajito[i])
            forcastini[i] = bajito[i][-1] * maxts / maxbaj
        parval = time_series_scaler.inverse_transform(time_series[-bunbunblack-3: -3].reshape(-1, 1))[:, 0]
        parpred = time_series_scaler.inverse_transform(forcastini.reshape(-1, 1))[:, 0]
        errpars.append(name_time_series+": "+str(mae(parpred, parval)))
        ser_val += parval
        pred += parpred
    if evo:
        return mae(pred, ser_val)
    else:
        return pred, ser_val, errpars, kip


def Duwanlift(kip, eme, numerodeamigos, nombre='bushlift'):
    pred, val, parcis, konf = vecilift(kip, eme, numerodeamigos, False)
    for parciales in parcis:
        print(parciales)
    print(mae(pred, val))
    print()
    print(kip)
    predmax = np.amax(pred)
    pred = pred * predmax
    val = val * predmax
    plt.figure()
    plt.plot(pred, marker='o', label='k-NN with lifting', color='C0')
    plt.plot(val, marker='o', label='Real values', color='C1')
    plt.xlabel('Forecast number')
    plt.ylabel('Scaled energy')
    plt.title('')
    plt.legend()
    plt.grid(which='both')
    plt.savefig(nombre+'.png')


cotillas = [(-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-2., 2.)]
eme = 61
amigs = 5
gongos = True
resultao = differential_evolution(vecilift, cotillas, args=(eme, amigs, gongos,), maxiter=55, popsize=8, disp=True, workers=-1, polish=False, updating='deferred')
Duwanlift(resultao.x, eme, amigs, 'wareware')
