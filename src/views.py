import base64
import io
from io import BytesIO

from django.contrib import messages
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, redirect

# Create your views here.
from django.template.loader import get_template
from django.urls import reverse
from django.utils.http import is_safe_url
from django.views.generic import View, TemplateView
from django.contrib.auth import (login as auth_login, logout as auth_logout,
                                 update_session_auth_hash)
import pydicom
from matplotlib.backends.backend_pdf import PdfPages
from  scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from xhtml2pdf import pisa

from src.forms import LoginForm


class LoginView(View):
    template = 'login.html'

    def get(self, request):
        form = LoginForm(request)
        next = 'home'
        return render(request, self.template, {'form': form, 'next': next})

    def post(self, request):
        form = LoginForm(request, data=request.POST)

        if form.is_valid():
            # Check where should the user be redirected
            next_redirect = request.POST.get('next', '')
            if not is_safe_url(url=next_redirect,
                               allowed_hosts=[request.get_host()]):
                next_redirect = reverse('/')

            auth_login(request, form.get_user())
            messages.info(request, 'Logged in as {}.'.format(request.user))
            return HttpResponseRedirect(next_redirect)

        return render(request, self.template, {'form': form})


class LogoutView(View):
    def get(self, request):
        if is_user_logged_in(request):
            auth_logout(request)
            # messages.info(request, "You have logged out.")

        return redirect('login')


def is_user_logged_in(request):
    """
    Returns True if the user is logged in. Returns False otherwise.
    """
    # If user not logged inform him about it
    if not request.user.is_authenticated:
        messages.error(request, 'You are not logged in.')
        return False

    return True

class HomeView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['screen'] = 'Lynx Analysis'
        context['type'] = 'report'
        return context

    def get(self, request, *args, **kwargs):
        context = self.get_context_data()
        data = {
            'offsetX': 0,
            'offsetY': 0,
            'negoffsetX': 0,
            'negoffsetY': 0,
            'rotation': 0
        }
        context['data'] = data
        return self.render_to_response(context)

    def post(self, request, *args, **kwargs):
        context = self.get_context_data()
        context['data'] = self.read_dicom()
        return self.render_to_response(context)

    def gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def read_dicom(self):
        measured_data = []
        positions = []
        uniformity = []
        sigma = []
        datatable = {}
        iCounter = 0
        files = self.request.FILES.getlist('files')
        offsetx = self.request.POST.get('offsetX')
        offsety = self.request.POST.get('offsetY')
        negoffsetx = self.request.POST.get('negoffsetX')
        negoffsety = self.request.POST.get('negoffsetY')
        rotation = self.request.POST.get('rotation_val')

        x = offsetx if float(offsetx) > 0 else negoffsetx
        y = offsety if float(offsety) > 0 else negoffsety
        iOffsetX = int(float(x) * 2)
        iOffsetY = int(float(y) * 2)
        irotation = float(rotation)
        iCounter = 0
        iNumFiles = len(files)
        neighborhood_size = 50
        threshold = 300
        iBuffer = 25
        npLayerData = np.zeros((iNumFiles, 16, 5))

        # Read dicom files
        for f in files:
            mSpotPositions = []
            mSigma = []

            sFilename = f
            npLynxData = pydicom.read_file(sFilename).pixel_array

            # Apply Lynx offset and rotation to data
            npLynxData = np.roll(npLynxData, iOffsetX, axis=1)  # apply x-offset
            npLynxData = np.roll(npLynxData, iOffsetY, axis=0)  # apply y-offset
            npLynxData = ndimage.rotate(npLynxData, irotation)
            # Calculate measured spot positions, sigma and uniformity
            data_max = filters.maximum_filter(npLynxData, neighborhood_size)
            maxima = (npLynxData == data_max)
            data_min = filters.minimum_filter(npLynxData, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []

            for dy, dx in slices:
                x_center = (dx.start + dx.stop - 1) / 2
                x.append(x_center)
                y_center = (dy.start + dy.stop - 1) / 2
                y.append(y_center)

                vYn = np.linspace(y_center - iBuffer, y_center + iBuffer, 51)
                vXn = np.linspace(x_center - iBuffer, x_center + iBuffer, 51)
                vProfilex = npLynxData[int(x_center - iBuffer):int(x_center + iBuffer + 1), int(y_center)]
                vProfiley = npLynxData[int(x_center), int(y_center - iBuffer):int(y_center + iBuffer + 1)]
                vOptx, vCovx = curve_fit(self.gauss, vXn, vProfilex, [300, x_center, 2])
                vOpty, vCovy = curve_fit(self.gauss, vYn, vProfiley, [300, y_center, 2])
                mSigma.append((vOptx[2] / 2, vOpty[2] / 2))  # yields sigma in mm

            npSigma = np.asarray(mSigma)  # np array of Sigma values

            x_pos = np.zeros(len(x))
            y_pos = np.zeros(len(y))

            for i in range(len(x)):
                x_pos[i] = float(x[i])
                y_pos[i] = float(y[i])
                mSpotPositions.append((x_pos[i], y_pos[i]))

            npSpotPositions = np.asarray(mSpotPositions)

            npUniformity = np.zeros((len(npSigma), 1))
            for i in range(len(npSigma)):
                npUniformity[i] = npSigma[i][0] / npSigma[i][1]  # np array of uniformity values

            npSpotData = np.concatenate((npSpotPositions, npSigma, npUniformity),
                                        axis=1)  # 16x5 numpy array containig measured spot x-y positions, x-y sigmas and uniformity (unsorted)

            # Sort spot data
            npSptPosMapd = npSpotPositions
            arrange = np.zeros((len(npSpotPositions)))
            for i in range(len(npSptPosMapd)):
                x = npSptPosMapd[:, 0][i]
                if 0 < x < 140:
                    x = 60
                if 80 < x < 300:
                    x = 220
                if 300 < x < 460:
                    x = 380
                if x > 460:
                    x = 540
                npSptPosMapd[:, 0][i] = x
                arrange[i] = (100 * npSptPosMapd[i][0] + npSptPosMapd[i][1])
            ind = np.argsort(arrange)  # numpy array of sorted indicies
            SpotData_sorted = []
            for i in ind:
                SpotData_sorted.append(npSpotData[
                                           i].tolist())  # 16x5 python list containig measured spot x-y positions, x-y sigmas and uniformity (sorted)

            # Clean spot data
            SpotData_sorted_clean = np.asarray(SpotData_sorted)

            for i in range(0, len(SpotData_sorted_clean) - 2):

                if i == len(SpotData_sorted_clean) - 1:
                    break
                if abs((SpotData_sorted_clean[i][0] - SpotData_sorted_clean[i + 1][0]) + (
                        SpotData_sorted_clean[i][1] - SpotData_sorted_clean[i + 1][1])) < 25:
                    SpotData_sorted_clean = np.delete(SpotData_sorted_clean, i, 0)

            SpotData_sorted_clean.tolist()
            npLayerData[iCounter] = np.asarray(
                SpotData_sorted_clean)  # 16 x 5 x #_of_layers numpy array containig all layer data

            # Create array of ideal spot positions.
            npIdealPositions = np.array(
                [[60, 60], [60, 220], [60, 380], [60, 540], [220, 60], [220, 220], [220, 380], [220, 540], [380, 60],
                 [380, 220], [380, 380], [380, 540], [540, 60], [540, 220], [540, 380], [540, 540]])

            # Find the difference of spot locations from ideal locations in pixels and in mm.
            npDeltaSpots = npLayerData[:, :, :2] - npIdealPositions
            npDeltaSpots_mm = npDeltaSpots / 2

            n = np.linspace(1, len(SpotData_sorted_clean[:, 0]), len(SpotData_sorted_clean[:, 1]))

            # ---- Generate Meansure Data Graph ---- #
            # fig = plt.figure(figsize=(15, 15))
            # ax1 = fig.subplots()
            fig, ax1 = plt.subplots(ncols=1, nrows=1)
            plt.title('Calculated (Black) & Ideal Positions (Grey)')
            xcoords = npIdealPositions[:, 0]
            for xc in xcoords:
                plt.axvline(x=xc, linewidth=0.25, color='0.75')
            ycoords = npIdealPositions[:, 1]
            for yc in ycoords:
                plt.axhline(y=yc, linewidth=0.25, color='0.75')
            plt.plot(npLayerData[iCounter, :, 0], npLayerData[iCounter, :, 1], 'k+', ms=5)
            plt.imshow(npLynxData)
            plt.colorbar()

            for i in range(len(n)):
                ax1.annotate(str(int(n[i])), xy=(npLayerData[:, :, 0][0][i] + 15, npLayerData[:, :, 1][0][i] + 15),
                             fontsize=15, color='White')

            graphic = self.get_image(plt)
            measured_data.append({'name': sFilename, 'image': graphic})
            # ---- End ---- #

            # fig = plt.figure(figsize=(13, 13))
            # # ax2 = fig.subplots()
            # ax2 = fig.add_subplot(222, xlim=[0, 17], ylim=[0, 10])
            fig, ax2 = plt.subplots(ncols=1, nrows=1)
            ax2.set_xlim([0, 17])
            ax2.set_ylim([0, 10])
            plt.title('Sigma')
            plt.xlabel('Spot Number')
            plt.ylabel('Sigma (mm)')
            plt.plot(n, npLayerData[iCounter, :, 2], 'rx', label='Sigma x')
            plt.plot(n, npLayerData[iCounter, :, 3], 'bx', label='Sigma y')
            plt.legend(loc='upper right', fontsize=12, numpoints=1)
            graphic = self.get_image(plt)
            sigma.append({'name': sFilename, 'image': graphic})

            # ---- Generate Position Graph ---- #
            # fig = plt.figure(figsize=(15, 15))
            # ax3 = fig.subplots()
            fig, ax3 = plt.subplots(ncols=1, nrows=1)
            ax3.set_xlim([0, 17])
            ax3.set_ylim([-4, 4])
            plt.title('Spot Position Deltas (mm)')
            plt.xlabel('Spot Number')
            plt.ylabel('Delta (mm)')
            plt.plot(n, npDeltaSpots_mm[iCounter, :, 0], 'rx', label='Delta x')
            plt.plot(n, npDeltaSpots_mm[iCounter, :, 1], 'bx', label='Delta y')
            plt.legend(loc='upper right', fontsize=12, numpoints=1)
            graphic = self.get_image(plt)
            positions.append({'name': sFilename, 'image': graphic})
            # ---- End ---- #

            # ---- Generate Uniformity Graph ---- #
            # fig = plt.figure(figsize=(15, 15))
            # ax2 = fig.subplots()
            fig, ax4 = plt.subplots(ncols=1, nrows=1)
            ax4.set_xlim([0, 17])
            ax4.set_ylim([0, 2])
            plt.title('Uniformity')
            plt.xlabel('Spot Number')
            plt.ylabel('Uniformity (%)')
            plt.plot(n, npLayerData[iCounter, :, 4], 'go')
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            uniformity.append({'name': sFilename, 'image': graphic})
            # ---- End ---- #

            # ---- Read Lync data table ---- #
            np.set_printoptions(precision=4)
            mean_delta_x = np.mean(npDeltaSpots_mm[iCounter, :, 0])
            mean_delta_y = np.mean(npDeltaSpots_mm[iCounter, :, 1])
            # max_delta_x = np.max(npDeltaSpots_mm[iCounter,:,0])
            # max_delta_y = np.max(npDeltaSpots_mm[iCounter,:,1])
            delta_x = npDeltaSpots_mm[iCounter, :, 0].tolist()
            delta_y = npDeltaSpots_mm[iCounter, :, 1].tolist()
            mean_sigma_x = np.mean(npLayerData[iCounter, :, 2])
            mean_sigma_y = np.mean(npLayerData[iCounter, :, 3])
            sigma_x = npLayerData[iCounter, :, 2].tolist()
            sigma_y = npLayerData[iCounter, :, 3].tolist()
            mean_uniformity = np.mean(npLayerData[iCounter, :, 4])

            max_x = np.max(npDeltaSpots_mm[iCounter, :, 0])
            min_x = np.min(npDeltaSpots_mm[iCounter, :, 0])
            if max_x > abs(min_x):
                max_delta_x = max_x
            else:
                max_delta_x = min_x

            max_y = np.max(npDeltaSpots_mm[iCounter, :, 1])
            min_y = np.min(npDeltaSpots_mm[iCounter, :, 1])
            if max_y > abs(min_y):
                max_delta_y = max_y
            else:
                max_delta_y = min_y

            mean_delta_x = '%.3f' % mean_delta_x
            mean_delta_y = '%.3f' % mean_delta_y
            max_delta_x = '%.3f' % max_delta_x
            max_delta_y = '%.3f' % max_delta_y
            delta_x = ['%.3f' % elem for elem in delta_x]
            delta_y = ['%.3f' % elem for elem in delta_y]
            mean_sigma_x = '%.3f' % mean_sigma_x
            mean_sigma_y = '%.3f' % mean_sigma_y
            sigma_x = ['%.3f' % elem for elem in sigma_x]
            sigma_y = ['%.3f' % elem for elem in sigma_y]
            mean_uniformity = '%.3f' % mean_uniformity
            datatable = {
                'mean_sigma_x': mean_sigma_x,
                'mean_sigma_y': mean_sigma_y,
                'mean_sigma_unif': mean_uniformity,
                'mean_delta_x': mean_delta_x,
                'mean_delta_y': mean_delta_y,
                'max_delta_x': max_delta_x,
                'max_delta_y': max_delta_y,
                'sigma_x': sigma_x,
                'sigma_y': sigma_y,
                'delta_x': delta_x,
                'delta_y': delta_y
            }
            # ---- End ---- #
            iCounter = iCounter + 1

        data = {
            'measured_data': measured_data,
            'sigma': sigma,
            'positions': positions,
            'uniformity': uniformity,
            'offsetX': offsetx,
            'offsetY': offsety,
            'negoffsetX': negoffsetx,
            'negoffsetY': negoffsety,
            'rotation': rotation,
            'files': files,
            'datatable': datatable
        }
        return data

    def get_image(self, plt):
        graphic = ''
        try:
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
        except:
            pass

        return graphic


class ExportView(View):
    def post(self, request, *args, **kwargs):
        data = self.read_dicom()
        return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

    def read_dicom(self):
        iCounter = 0
        files = self.request.FILES.getlist('files')
        offsetx = self.request.POST.get('offsetX')
        offsety = self.request.POST.get('offsetY')
        negoffsetx = self.request.POST.get('negoffsetX')
        negoffsety = self.request.POST.get('negoffsetY')
        rotation = self.request.POST.get('rotation_val')

        x = offsetx if float(offsetx) > 0 else negoffsetx
        y = offsety if float(offsety) > 0 else negoffsety

        iOffsetX = int(float(x) * 2)
        iOffsetY = int(float(y) * 2)
        irotation = float(rotation)
        iCounter = 0
        iNumFiles = len(files)
        neighborhood_size = 50
        threshold = 300
        iBuffer = 25
        npLayerData = np.zeros((iNumFiles, 16, 5))
        for f in files:
            mSpotPositions = []
            mSigma = []

            sFilename = f
            npLynxData = pydicom.read_file(sFilename).pixel_array

            # Apply Lynx offset and rotation to data
            npLynxData = np.roll(npLynxData, iOffsetX, axis=1)  # apply x-offset
            npLynxData = np.roll(npLynxData, iOffsetY, axis=0)  # apply y-offset
            npLynxData = ndimage.rotate(npLynxData, irotation)
            # Calculate measured spot positions, sigma and uniformity
            data_max = filters.maximum_filter(npLynxData, neighborhood_size)
            maxima = (npLynxData == data_max)
            data_min = filters.minimum_filter(npLynxData, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []

            for dy, dx in slices:
                x_center = (dx.start + dx.stop - 1) / 2
                x.append(x_center)
                y_center = (dy.start + dy.stop - 1) / 2
                y.append(y_center)

                vYn = np.linspace(y_center - iBuffer, y_center + iBuffer, 51)
                vXn = np.linspace(x_center - iBuffer, x_center + iBuffer, 51)
                vProfilex = npLynxData[int(x_center - iBuffer):int(x_center + iBuffer + 1), int(y_center)]
                vProfiley = npLynxData[int(x_center), int(y_center - iBuffer):int(y_center + iBuffer + 1)]
                vOptx, vCovx = curve_fit(HomeView().gauss, vXn, vProfilex, [300, x_center, 2])
                vOpty, vCovy = curve_fit(HomeView().gauss, vYn, vProfiley, [300, y_center, 2])
                mSigma.append((vOptx[2] / 2, vOpty[2] / 2))  # yields sigma in mm

            npSigma = np.asarray(mSigma)  # np array of Sigma values

            x_pos = np.zeros(len(x))
            y_pos = np.zeros(len(y))

            for i in range(len(x)):
                x_pos[i] = float(x[i])
                y_pos[i] = float(y[i])
                mSpotPositions.append((x_pos[i], y_pos[i]))

            npSpotPositions = np.asarray(mSpotPositions)

            npUniformity = np.zeros((len(npSigma), 1))
            for i in range(len(npSigma)):
                npUniformity[i] = npSigma[i][0] / npSigma[i][1]  # np array of uniformity values

            npSpotData = np.concatenate((npSpotPositions, npSigma, npUniformity),
                                        axis=1)  # 16x5 numpy array containig measured spot x-y positions, x-y sigmas and uniformity (unsorted)

            # Sort spot data
            npSptPosMapd = npSpotPositions
            arrange = np.zeros((len(npSpotPositions)))
            for i in range(len(npSptPosMapd)):
                x = npSptPosMapd[:, 0][i]
                if 0 < x < 140:
                    x = 60
                if 80 < x < 300:
                    x = 220
                if 300 < x < 460:
                    x = 380
                if x > 460:
                    x = 540
                npSptPosMapd[:, 0][i] = x
                arrange[i] = (100 * npSptPosMapd[i][0] + npSptPosMapd[i][1])
            ind = np.argsort(arrange)  # numpy array of sorted indicies
            SpotData_sorted = []
            for i in ind:
                SpotData_sorted.append(npSpotData[
                                           i].tolist())  # 16x5 python list containig measured spot x-y positions, x-y sigmas and uniformity (sorted)

            # Clean spot data
            SpotData_sorted_clean = np.asarray(SpotData_sorted)

            for i in range(0, len(SpotData_sorted_clean) - 2):

                if i == len(SpotData_sorted_clean) - 1:
                    break
                if abs((SpotData_sorted_clean[i][0] - SpotData_sorted_clean[i + 1][0]) + (
                        SpotData_sorted_clean[i][1] - SpotData_sorted_clean[i + 1][1])) < 25:
                    SpotData_sorted_clean = np.delete(SpotData_sorted_clean, i, 0)

            SpotData_sorted_clean.tolist()
            npLayerData[iCounter] = np.asarray(
                SpotData_sorted_clean)  # 16 x 5 x #_of_layers numpy array containig all layer data

            # Create array of ideal spot positions.
            npIdealPositions = np.array(
                [[60, 60], [60, 220], [60, 380], [60, 540], [220, 60], [220, 220], [220, 380], [220, 540], [380, 60],
                 [380, 220], [380, 380], [380, 540], [540, 60], [540, 220], [540, 380], [540, 540]])

            # Find the difference of spot locations from ideal locations in pixels and in mm.
            npDeltaSpots = npLayerData[:, :, :2] - npIdealPositions
            npDeltaSpots_mm = npDeltaSpots / 2

            # Generate and print plots to PDF
            n = np.linspace(1, len(SpotData_sorted_clean[:, 0]), len(SpotData_sorted_clean[:, 1]))
            fig = plt.figure(figsize=(15, 15))

            ax1 = fig.add_subplot(221)
            plt.title('Calculated (Black) & Ideal Positions (Grey)')
            # plt.plot(npIdealPositions[:,0], npIdealPositions[:,1], 'wx')
            xcoords = npIdealPositions[:, 0]
            for xc in xcoords:
                plt.axvline(x=xc, linewidth=0.25, color='0.75')
            ycoords = npIdealPositions[:, 1]
            for yc in ycoords:
                plt.axhline(y=yc, linewidth=0.25, color='0.75')
            plt.plot(npLayerData[iCounter, :, 0], npLayerData[iCounter, :, 1], 'k+', ms=5)
            plt.imshow(npLynxData)
            plt.colorbar()
            for i in range(len(n)):
                ax1.annotate(str(int(n[i])), xy=(npLayerData[:, :, 0][0][i] + 15, npLayerData[:, :, 1][0][i] + 15),
                             fontsize=15, color='White')

            ax2 = fig.add_subplot(222, xlim=[0, 17], ylim=[0, 10])
            plt.title('Sigma')
            plt.xlabel('Spot Number')
            plt.ylabel('Sigma (mm)')
            plt.plot(n, npLayerData[iCounter, :, 2], 'rx', label='Sigma x')
            plt.plot(n, npLayerData[iCounter, :, 3], 'bx', label='Sigma y')
            plt.legend(loc='upper right', fontsize=12, numpoints=1)

            ax3 = fig.add_subplot(223, xlim=[0, 17], ylim=[-4, 4])
            plt.title('Spot Position Deltas (mm)')
            plt.xlabel('Spot Number')
            plt.ylabel('Delta (mm)')
            plt.plot(n, npDeltaSpots_mm[iCounter, :, 0], 'rx', label='Delta x')
            plt.plot(n, npDeltaSpots_mm[iCounter, :, 1], 'bx', label='Delta y')
            plt.legend(loc='upper right', fontsize=12, numpoints=1)

            ax2 = fig.add_subplot(224, xlim=[0, 17], ylim=[0, 2])
            plt.title('Uniformity')
            plt.xlabel('Spot Number')
            plt.ylabel('Uniformity (%)')
            plt.plot(n, npLayerData[iCounter, :, 4], 'go')
            plt.savefig('Layer ' + str(iCounter + 1) + '.pdf')

            # Output data to excel 'Lynx.csv
            np.set_printoptions(precision=4)
            mean_delta_x = np.mean(npDeltaSpots_mm[iCounter, :, 0])
            mean_delta_y = np.mean(npDeltaSpots_mm[iCounter, :, 1])
            # max_delta_x = np.max(npDeltaSpots_mm[iCounter,:,0])
            # max_delta_y = np.max(npDeltaSpots_mm[iCounter,:,1])
            delta_x = npDeltaSpots_mm[iCounter, :, 0].tolist()
            delta_y = npDeltaSpots_mm[iCounter, :, 1].tolist()
            mean_sigma_x = np.mean(npLayerData[iCounter, :, 2])
            mean_sigma_y = np.mean(npLayerData[iCounter, :, 3])
            sigma_x = npLayerData[iCounter, :, 2].tolist()
            sigma_y = npLayerData[iCounter, :, 3].tolist()
            mean_uniformity = np.mean(npLayerData[iCounter, :, 4])

            max_x = np.max(npDeltaSpots_mm[iCounter, :, 0])
            min_x = np.min(npDeltaSpots_mm[iCounter, :, 0])
            if max_x > abs(min_x):
                max_delta_x = max_x
            else:
                max_delta_x = min_x

            max_y = np.max(npDeltaSpots_mm[iCounter, :, 1])
            min_y = np.min(npDeltaSpots_mm[iCounter, :, 1])
            if max_y > abs(min_y):
                max_delta_y = max_y
            else:
                max_delta_y = min_y

            mean_delta_x = '%.3f' % mean_delta_x
            mean_delta_y = '%.3f' % mean_delta_y
            max_delta_x = '%.3f' % max_delta_x
            max_delta_y = '%.3f' % max_delta_y
            delta_x = ['%.3f' % elem for elem in delta_x]
            delta_y = ['%.3f' % elem for elem in delta_y]
            mean_sigma_x = '%.3f' % mean_sigma_x
            mean_sigma_y = '%.3f' % mean_sigma_y
            sigma_x = ['%.3f' % elem for elem in sigma_x]
            sigma_y = ['%.3f' % elem for elem in sigma_y]
            mean_uniformity = '%.3f' % mean_uniformity

            with open('Lynx.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Layer ' + str(iCounter + 1), 'MEAN SIGMA X', mean_sigma_x, 'MEAN SIGMA Y', mean_sigma_y,
                                 'MEAN SIGMA UNIFORMITY', mean_uniformity, 'MEAN DELTA X', mean_delta_x,
                                 'MEAN DELTA Y', mean_delta_y, 'MAX DELTA X', max_delta_x, 'MAX DELTA Y', max_delta_y,
                                 'SIGMA X', sigma_x, 'SIGMA Y', sigma_y, 'DELTA X', delta_x, 'DELTA Y', delta_y])

            iCounter = iCounter + 1


class PrintView(View):
    def post(self, request, *args, **kwargs):
        template = get_template('print.html')
        data = self.read_dicom()
        html = template.render({'data':data})
        results = BytesIO()
        pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), results)
        if not pdf.err:
            return HttpResponse(results.getvalue(), content_type='application/pdf')
        return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

    def read_dicom(self):
        data = []
        datatable = {}
        iCounter = 0
        files = self.request.FILES.getlist('files')
        offsetx = self.request.POST.get('offsetX')
        offsety = self.request.POST.get('offsetY')
        negoffsetx = self.request.POST.get('negoffsetX')
        negoffsety = self.request.POST.get('negoffsetY')
        rotation = self.request.POST.get('rotation_val')

        x = offsetx if float(offsetx) > 0 else negoffsetx
        y = offsety if float(offsety) > 0 else negoffsety

        iOffsetX = int(float(x) * 2)
        iOffsetY = int(float(y) * 2)
        irotation = float(rotation)
        iCounter = 0
        iNumFiles = len(files)
        neighborhood_size = 50
        threshold = 300
        iBuffer = 25
        npLayerData = np.zeros((iNumFiles, 16, 5))

        # Read dicom files
        for f in files:
            mSpotPositions = []
            mSigma = []

            sFilename = f
            npLynxData = pydicom.read_file(sFilename).pixel_array

            # Apply Lynx offset and rotation to data
            npLynxData = np.roll(npLynxData, iOffsetX, axis=1)  # apply x-offset
            npLynxData = np.roll(npLynxData, iOffsetY, axis=0)  # apply y-offset
            npLynxData = ndimage.rotate(npLynxData, irotation)
            # Calculate measured spot positions, sigma and uniformity
            data_max = filters.maximum_filter(npLynxData, neighborhood_size)
            maxima = (npLynxData == data_max)
            data_min = filters.minimum_filter(npLynxData, neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []

            for dy, dx in slices:
                x_center = (dx.start + dx.stop - 1) / 2
                x.append(x_center)
                y_center = (dy.start + dy.stop - 1) / 2
                y.append(y_center)

                vYn = np.linspace(y_center - iBuffer, y_center + iBuffer, 51)
                vXn = np.linspace(x_center - iBuffer, x_center + iBuffer, 51)
                vProfilex = npLynxData[int(x_center - iBuffer):int(x_center + iBuffer + 1), int(y_center)]
                vProfiley = npLynxData[int(x_center), int(y_center - iBuffer):int(y_center + iBuffer + 1)]
                vOptx, vCovx = curve_fit(HomeView().gauss, vXn, vProfilex, [300, x_center, 2])
                vOpty, vCovy = curve_fit(HomeView().gauss, vYn, vProfiley, [300, y_center, 2])
                mSigma.append((vOptx[2] / 2, vOpty[2] / 2))  # yields sigma in mm

            npSigma = np.asarray(mSigma)  # np array of Sigma values

            x_pos = np.zeros(len(x))
            y_pos = np.zeros(len(y))

            for i in range(len(x)):
                x_pos[i] = float(x[i])
                y_pos[i] = float(y[i])
                mSpotPositions.append((x_pos[i], y_pos[i]))

            npSpotPositions = np.asarray(mSpotPositions)

            npUniformity = np.zeros((len(npSigma), 1))
            for i in range(len(npSigma)):
                npUniformity[i] = npSigma[i][0] / npSigma[i][1]  # np array of uniformity values

            npSpotData = np.concatenate((npSpotPositions, npSigma, npUniformity),
                                        axis=1)  # 16x5 numpy array containig measured spot x-y positions, x-y sigmas and uniformity (unsorted)

            # Sort spot data
            npSptPosMapd = npSpotPositions
            arrange = np.zeros((len(npSpotPositions)))
            for i in range(len(npSptPosMapd)):
                x = npSptPosMapd[:, 0][i]
                if 0 < x < 140:
                    x = 60
                if 80 < x < 300:
                    x = 220
                if 300 < x < 460:
                    x = 380
                if x > 460:
                    x = 540
                npSptPosMapd[:, 0][i] = x
                arrange[i] = (100 * npSptPosMapd[i][0] + npSptPosMapd[i][1])
            ind = np.argsort(arrange)  # numpy array of sorted indicies
            SpotData_sorted = []
            for i in ind:
                SpotData_sorted.append(npSpotData[
                                           i].tolist())  # 16x5 python list containig measured spot x-y positions, x-y sigmas and uniformity (sorted)

            # Clean spot data
            SpotData_sorted_clean = np.asarray(SpotData_sorted)

            for i in range(0, len(SpotData_sorted_clean) - 2):

                if i == len(SpotData_sorted_clean) - 1:
                    break
                if abs((SpotData_sorted_clean[i][0] - SpotData_sorted_clean[i + 1][0]) + (
                        SpotData_sorted_clean[i][1] - SpotData_sorted_clean[i + 1][1])) < 25:
                    SpotData_sorted_clean = np.delete(SpotData_sorted_clean, i, 0)

            SpotData_sorted_clean.tolist()
            npLayerData[iCounter] = np.asarray(
                SpotData_sorted_clean)  # 16 x 5 x #_of_layers numpy array containig all layer data

            # Create array of ideal spot positions.
            npIdealPositions = np.array(
                [[60, 60], [60, 220], [60, 380], [60, 540], [220, 60], [220, 220], [220, 380], [220, 540], [380, 60],
                 [380, 220], [380, 380], [380, 540], [540, 60], [540, 220], [540, 380], [540, 540]])

            # Find the difference of spot locations from ideal locations in pixels and in mm.
            npDeltaSpots = npLayerData[:, :, :2] - npIdealPositions
            npDeltaSpots_mm = npDeltaSpots / 2

            n = np.linspace(1, len(SpotData_sorted_clean[:, 0]), len(SpotData_sorted_clean[:, 1]))

            # ---- Generate Meansure Data Graph ---- #
            # fig = plt.figure(figsize=(15, 15))
            # ax1 = fig.subplots()
            fig, ax1 = plt.subplots(ncols=1, nrows=1)
            plt.title('Calculated (Black) & Ideal Positions (Grey)')
            xcoords = npIdealPositions[:, 0]
            for xc in xcoords:
                plt.axvline(x=xc, linewidth=0.25, color='0.75')
            ycoords = npIdealPositions[:, 1]
            for yc in ycoords:
                plt.axhline(y=yc, linewidth=0.25, color='0.75')
            plt.plot(npLayerData[iCounter, :, 0], npLayerData[iCounter, :, 1], 'k+', ms=5)
            plt.imshow(npLynxData)
            plt.colorbar()

            for i in range(len(n)):
                ax1.annotate(str(int(n[i])), xy=(npLayerData[:, :, 0][0][i] + 15, npLayerData[:, :, 1][0][i] + 15),
                             fontsize=15, color='White')

            graphic = HomeView().get_image(plt)
            measured_data = {'name': sFilename, 'image': graphic}
            # ---- End ---- #

            # fig = plt.figure(figsize=(13, 13))
            # # ax2 = fig.subplots()
            # ax2 = fig.add_subplot(222, xlim=[0, 17], ylim=[0, 10])
            fig, ax2 = plt.subplots(ncols=1, nrows=1)
            ax2.set_xlim([0, 17])
            ax2.set_ylim([0, 10])
            plt.title('Sigma')
            plt.xlabel('Spot Number')
            plt.ylabel('Sigma (mm)')
            plt.plot(n, npLayerData[iCounter, :, 2], 'rx', label='Sigma x')
            plt.plot(n, npLayerData[iCounter, :, 3], 'bx', label='Sigma y')
            plt.legend(loc='upper right', fontsize=12, numpoints=1)
            graphic = HomeView().get_image(plt)
            sigma = {'name': sFilename, 'image': graphic}

            # ---- Generate Position Graph ---- #
            # fig = plt.figure(figsize=(15, 15))
            # ax3 = fig.subplots()
            fig, ax3 = plt.subplots(ncols=1, nrows=1)
            ax3.set_xlim([0, 17])
            ax3.set_ylim([-4, 4])
            plt.title('Spot Position Deltas (mm)')
            plt.xlabel('Spot Number')
            plt.ylabel('Delta (mm)')
            plt.plot(n, npDeltaSpots_mm[iCounter, :, 0], 'rx', label='Delta x')
            plt.plot(n, npDeltaSpots_mm[iCounter, :, 1], 'bx', label='Delta y')
            plt.legend(loc='upper right', fontsize=12, numpoints=1)
            graphic = HomeView().get_image(plt)
            positions = {'name': sFilename, 'image': graphic}
            # ---- End ---- #

            # ---- Generate Uniformity Graph ---- #
            # fig = plt.figure(figsize=(15, 15))
            # ax2 = fig.subplots()
            fig, ax4 = plt.subplots(ncols=1, nrows=1)
            ax4.set_xlim([0, 17])
            ax4.set_ylim([0, 2])
            plt.title('Uniformity')
            plt.xlabel('Spot Number')
            plt.ylabel('Uniformity (%)')
            plt.plot(n, npLayerData[iCounter, :, 4], 'go')
            graphic = HomeView().get_image(plt)
            uniformity = {'name': sFilename, 'image': graphic}
            # ---- End ---- #

            # ---- Read Lync data table ---- #
            np.set_printoptions(precision=4)
            mean_delta_x = np.mean(npDeltaSpots_mm[iCounter, :, 0])
            mean_delta_y = np.mean(npDeltaSpots_mm[iCounter, :, 1])
            # max_delta_x = np.max(npDeltaSpots_mm[iCounter,:,0])
            # max_delta_y = np.max(npDeltaSpots_mm[iCounter,:,1])
            delta_x = npDeltaSpots_mm[iCounter, :, 0].tolist()
            delta_y = npDeltaSpots_mm[iCounter, :, 1].tolist()
            mean_sigma_x = np.mean(npLayerData[iCounter, :, 2])
            mean_sigma_y = np.mean(npLayerData[iCounter, :, 3])
            sigma_x = npLayerData[iCounter, :, 2].tolist()
            sigma_y = npLayerData[iCounter, :, 3].tolist()
            mean_uniformity = np.mean(npLayerData[iCounter, :, 4])

            max_x = np.max(npDeltaSpots_mm[iCounter, :, 0])
            min_x = np.min(npDeltaSpots_mm[iCounter, :, 0])
            if max_x > abs(min_x):
                max_delta_x = max_x
            else:
                max_delta_x = min_x

            max_y = np.max(npDeltaSpots_mm[iCounter, :, 1])
            min_y = np.min(npDeltaSpots_mm[iCounter, :, 1])
            if max_y > abs(min_y):
                max_delta_y = max_y
            else:
                max_delta_y = min_y

            mean_delta_x = '%.3f' % mean_delta_x
            mean_delta_y = '%.3f' % mean_delta_y
            max_delta_x = '%.3f' % max_delta_x
            max_delta_y = '%.3f' % max_delta_y
            delta_x = ['%.3f' % elem for elem in delta_x]
            delta_y = ['%.3f' % elem for elem in delta_y]
            mean_sigma_x = '%.3f' % mean_sigma_x
            mean_sigma_y = '%.3f' % mean_sigma_y
            sigma_x = ['%.3f' % elem for elem in sigma_x]
            sigma_y = ['%.3f' % elem for elem in sigma_y]
            mean_uniformity = '%.3f' % mean_uniformity
            datatable = {
                'mean_sigma_x': mean_sigma_x,
                'mean_sigma_y': mean_sigma_y,
                'mean_sigma_unif': mean_uniformity,
                'mean_delta_x': mean_delta_x,
                'mean_delta_y': mean_delta_y,
                'max_delta_x': max_delta_x,
                'max_delta_y': max_delta_y,
                'sigma_x': sigma_x,
                'sigma_y': sigma_y,
                'delta_x': delta_x,
                'delta_y': delta_y
            }
            # ---- End ---- #
            iCounter = iCounter + 1

            data.append({
                'measured_data': measured_data,
                'sigma': sigma,
                'positions': positions,
                'uniformity': uniformity,
                'datatable': datatable
            })
        return data