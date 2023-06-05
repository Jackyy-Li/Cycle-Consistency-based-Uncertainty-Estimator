from distutils.log import error
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.ndimage
import scipy.stats
import sklearn.metrics
import pandas as pd


wk_path = r'.\output'
wk_paths = glob.glob(os.path.join(wk_path, '*'))
FIT_RANGE = 3


def RMSE(im, yy):
    if im.shape != yy.shape:
        yy = scipy.ndimage.zoom(yy, (im.shape[0]/yy.shape[0], im.shape[1]/yy.shape[1], 1))
    return np.sqrt(np.mean(np.square(im - yy)))


def Taylor_1st_order(n, *params):
    l, e, eps = params
    return l**n * max(0,e) + max(0,eps)


def get_init_p(v_list):
    # get initial params
    p0 = np.zeros(3)
    p0[2] = v_list[-1]  # converging value as eps
    p0[1] = max(0, v_list[0] - p0[2])  # intercept as e
    p0[0] = max(1, (v_list[1] - p0[2]) / (v_list[0] - p0[2]))
    return p0


####################
# Save format:
# x is the input, y0 is the prediction, y is the ground truth
# xx_list: [x, x1, x2, ...], (N+1) * Hx * Wx * C, input images of each cycle
# im_list: [y0, y1, y2, ...], N * Hy * Wy * C, output images of each cycle
# yy: y, Hy * Wy * C, ground truth
####################


for wk_path in wk_paths:
    output_files = glob.glob(os.path.join(glob.escape(wk_path), '*.mat'))
    rmse_diff_mat = []  # y differential matrix
    rmse_fit_list = []  # uncertainty estimate list
    xrmse_fit_list = []  # uncertainty estimate list from x
    bias_fit_list = []  # bias estimate list
    xbias_fit_list = []  # bias estimate list from x
    robust_list = []  # k' list
    fit_wellness = []  # R2 list
    xrmse_list = []  # e0 list from x, difference between x and x1
    rmse_list = []  # e0 list
    for f in output_files:
        tmp = sio.loadmat(f)
        # calculate GT
        yy = tmp['yy']
        if yy.dtype == 'uint8':
            yy = yy.astype('float32') / 255
        # if yy.shape[0] != 3 & yy.shape[0] != 4:
        #     yy = yy.transpose([2,0,1])
        # calculate y0
        xx = tmp['xx_list'].squeeze()
        if xx.dtype == 'uint8':
            xx = xx.astype('float32') / 255
        im = tmp['im_list'].squeeze()
        if im.dtype == 'uint8':
            im = im.astype('float32') / 255
        x = xx[0,...]
        x1 = xx[1,...]
        im0 = im[0,...]
        # calculate e0
        xrmse = RMSE(x, x1)
        rmse = RMSE(im0, yy)
        xrmse_list.append(xrmse)
        rmse_list.append(rmse)
        # calculate y differential
        xx_diff = np.diff(xx[1:], axis=0)  # exclude x
        im_diff = np.diff(im, axis=0)
        # im_diff = im[1:,...] - im0
        xrmse_diff = np.sqrt(np.mean(np.square(xx_diff), axis=(1,2,3)))
        rmse_diff = np.sqrt(np.mean(np.square(np.abs(im_diff)), axis=(1,2,3)))

        xp0 = get_init_p(xrmse_diff)
        xpopt, _ = scipy.optimize.curve_fit(Taylor_1st_order, np.arange(1,1+FIT_RANGE), xrmse_diff[:FIT_RANGE], p0=xp0, maxfev=100000)
        xrmse_pred = Taylor_1st_order(np.arange(1,1+FIT_RANGE), *xpopt)
        xrmse_fit_list.append(max(0, xpopt[1]))
        xbias_fit_list.append(max(0, xpopt[2]))

        p0 = get_init_p(rmse_diff)
        popt, _ = scipy.optimize.curve_fit(Taylor_1st_order, np.arange(1,1+FIT_RANGE), rmse_diff[:FIT_RANGE], p0=p0, maxfev=100000)
        rmse_pred = Taylor_1st_order(np.arange(1,1+FIT_RANGE), *popt)
        r2 = sklearn.metrics.r2_score(rmse_diff[:FIT_RANGE], rmse_pred)
        if popt[1] >= 1:
            print('DEBUG')
        rmse_fit_list.append(max(0, popt[1]))
        bias_fit_list.append(max(0, popt[2]))
        robust_list.append(popt[0])
        fit_wellness.append(r2)

    print('length of output files: {}'.format(len(output_files)))
    print('length of rmse_list_fit_list: {}'.format(len(xrmse_list)))

    # check fitting wellness
    if np.mean(fit_wellness) < 0.8:
        print('Fitting wellness is too low: {}. Consider changing fitting initial values'.format(np.mean(fit_wellness)))

    print("length before outlier removal: {}".format(len(rmse_fit_list)))


    # remove outliers based on R2, setting the thredshold to be 0.5
    non_outliers = np.where(np.array(fit_wellness) > 0.5)[0]
    rmse_fit_list = list(np.array(rmse_fit_list)[non_outliers])
    xrmse_fit_list = list(np.array(xrmse_fit_list)[non_outliers])
    rmse_list = list(np.array(rmse_list)[non_outliers])
    xrmse_list = list(np.array(xrmse_list)[non_outliers])
    bias_fit_list = list(np.array(bias_fit_list)[non_outliers])
    xbias_fit_list = list(np.array(xbias_fit_list)[non_outliers])

    print("length after r2 outlier removal: {}".format(len(rmse_fit_list)))

    # remove outliers
    rmse_fit_z = (rmse_fit_list - np.mean(rmse_fit_list)) / np.std(rmse_fit_list)
    outlier_idx = np.where(np.abs(rmse_fit_z) > 3)[0]
    while len(outlier_idx) > 0:
        for id in reversed(outlier_idx):
            del rmse_fit_list[id]
            del xrmse_fit_list[id]
            del rmse_list[id]
            del xrmse_list[id]
            del bias_fit_list[id]
            del xbias_fit_list[id]
        rmse_fit_z = (rmse_fit_list - np.mean(rmse_fit_list)) / np.std(rmse_fit_list)
        outlier_idx = np.where(np.abs(rmse_fit_z) > 3)[0]

    # visualization
    f1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,8), dpi=300)
    ax1.plot(rmse_fit_list, rmse_list, 'bo')
    ax2.plot(xrmse_fit_list, rmse_list, 'ro')
    ax3.plot(xrmse_fit_list, xrmse_list, 'go')
    ax4.plot(xrmse_list, rmse_list, 'ko')
    f1.savefig(os.path.join(wk_path, 'RMSE_fit%d_alt.png'%FIT_RANGE))

    #save 4 vectors as dataframe to csv file
    if (wk_path.split("/")[-1].split("_")[-1] == 'n=0.00'):
            class_val = 0
    else:
            class_val = 1

    df = pd.DataFrame({
        'rmse_fit_list':rmse_fit_list,
        'xrmse_fit_list':xrmse_fit_list,
        'rmse_list':rmse_list,
        "xrmse_list":xrmse_list,
        'bias_list':bias_fit_list,
        'xbias_list':xbias_fit_list,
        #class 1 if any kernel or noise, 0 otherwise
        'class':class_val*np.ones(len(rmse_fit_list))
    })
    df.to_csv(os.path.join(wk_path, 'RMSE_fit%d_alt.csv'%FIT_RANGE), index=False)

    # record & print
    with open(os.path.join(wk_path, 'metrics%d_alt.txt'%FIT_RANGE), 'w') as f:
        # f.write('RMSE fitting wellness: %.4f\n'%error_fit_ls.rvalue**2)
        f.write('Average absolute error: %.4e\n'%np.mean(rmse_list))
        f.write('Average predicted error: %.4e\n'%np.mean(rmse_fit_list))
        f.write('Average x-predicted error: %.4e\n'%np.mean(xrmse_fit_list))
        f.write('Average x-absolute error: %.4e\n'%np.mean(xrmse_list))
        f.write('Average instability: %.4e\n'%np.mean(robust_list))
        f.write('Fitting wellness (R^2): %.4f\n'%np.mean(fit_wellness))

    print('Average absolute error: %.4e'%np.mean(rmse_list))
    print('Average fitting error: %.4e'%np.mean(rmse_fit_list))
    print('Average x-predicted error: %.4e'%np.mean(xrmse_fit_list))
    print('Average x-absolute error: %.4e'%np.mean(xrmse_list))
    print('Average instability: %.4e'%np.mean(robust_list))
    print('Fitting wellness (R^2): %.4f'%np.mean(fit_wellness))
    print('length of each list: ',str(len(rmse_fit_list)))