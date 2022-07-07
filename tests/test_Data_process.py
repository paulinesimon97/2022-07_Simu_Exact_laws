import os

from exact_laws.preprocessing.reformating_OCA_file import *
from exact_laws.preprocessing.scan_file import bin_an_array
import pytest
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


class TestVerifFileExistence:
    """voir contenu du fichier output_process_[...].txt."""
    pass


class TestExtractSimuParam:
    """voir contenu du fichier output_process_[...].txt."""
    pass


class TestExtractQuantities:
    """voir contenu du fichier output_process_[...].txt."""
    pass


@pytest.fixture
def init_inputdic():
    inputdic = {}
    inputdic['BG17'] = False
    inputdic['BG17Hall'] = False
    inputdic['SS22I'] = False
    inputdic['SS22IGyr'] = False
    inputdic['SS22IHall'] = False
    inputdic['SS22C'] = False
    inputdic['SS22CIso'] = False
    inputdic['SS22CGyr'] = False
    inputdic['SS22CHall'] = False
    inputdic['SS21C'] = False
    inputdic['SS21CIso'] = False
    inputdic['SS21CHom'] = False
    return inputdic


class TestCreatDicWant:
    def test_BG17(self, init_inputdic):
        init_inputdic['BG17'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': True, 'gradv': False, 'delv': False, \
                        'rho': False, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': True, 'b': False, 'delb': False, 'Ij': True, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_BG17Hall(self, init_inputdic):
        init_inputdic['BG17Hall'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': False, 'w': False, 'gradv': False, 'delv': False, \
                        'rho': False, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': True, 'b': False, 'delb': False, 'Ij': True, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22I(self, init_inputdic):
        init_inputdic['SS22I'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': False, 'gradv': False, 'delv': False, \
                        'rho': False, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': True, 'b': False, 'delb': False, 'Ij': False, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22IGyr(self, init_inputdic):
        init_inputdic['SS22IGyr'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': False, 'w': False, 'gradv': True, 'delv': False, \
                        'rho': False, 'delrho': False, \
                        'Igyrp': True, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': True, 'b': False, 'delb': False, 'Ij': False, 'j': False, 'delj': False, 'Ipm': True,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22IHall(self, init_inputdic):
        init_inputdic['SS22IHall'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': False, 'w': False, 'gradv': False, 'delv': False, \
                        'rho': False, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': True, 'b': False, 'delb': False, 'Ij': True, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22C(self, init_inputdic):
        init_inputdic['SS22C'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': False, 'gradv': False, 'delv': True, \
                        'rho': True, 'delrho': True, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': False, 'b': True, 'delb': True, 'Ij': False, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': True}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22CIso(self, init_inputdic):
        init_inputdic['SS22CIso'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': False, 'gradv': False, 'delv': True, \
                        'rho': True, 'delrho': True, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': True, 'isou': True, 'delisou': False, \
                        'Ib': False, 'b': False, 'delb': False, 'Ij': False, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22CGyr(self, init_inputdic):
        init_inputdic['SS22CGyr'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': False, 'gradv': True, 'delv': True, \
                        'rho': True, 'delrho': True, \
                        'Igyrp': False, 'gyrp': True, 'gyru': True, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': False, 'b': True, 'delb': False, 'Ij': False, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': True}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS22CHall(self, init_inputdic):
        init_inputdic['SS22CHall'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': False, 'w': False, 'gradv': False, 'delv': False, \
                        'rho': True, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': False, 'b': True, 'delb': True, 'Ij': False, 'j': True, 'delj': True, 'Ipm': False,
                        'pm': False}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS21C(self, init_inputdic):
        init_inputdic['SS21C'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': False, 'gradv': False, 'delv': True, \
                        'rho': True, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': False, 'isou': False, 'delisou': False, \
                        'Ib': False, 'b': True, 'delb': True, 'Ij': False, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': True}
        assert dic_want == expected_dic, f"error on the return dic"

    def test_SS21CIso(self, init_inputdic):
        init_inputdic['SS21CIso'] = True
        dic_want = creat_dic_want(init_inputdic)
        expected_dic = {'v': True, 'w': False, 'gradv': False, 'delv': True, \
                        'rho': True, 'delrho': False, \
                        'Igyrp': False, 'gyrp': False, 'gyru': False, 'isop': True, 'isou': True, 'delisou': True, \
                        'Ib': False, 'b': False, 'delb': False, 'Ij': False, 'j': False, 'delj': False, 'Ipm': False,
                        'pm': True}
        assert dic_want == expected_dic, f"error on the return dic"


@pytest.fixture
def init_arg_fv():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['vx'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['vy'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['vz'] = np.array(
        [3 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['wx'] = np.array(
        [3 * sinx[i] * cosx[j] * sinx[k] - 2 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_quant['wy'] = np.array(
        [sinx[i] * sinx[j] * cosx[k] - 3 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for
         k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_quant['wz'] = np.array(
        [2 * cosx[i] * sinx[j] * sinx[k] - sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for
         k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_quant['dxvx'] = np.array(
        [cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dyvx'] = np.array(
        [sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dzvx'] = np.array(
        [sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dxvy'] = np.array(
        [2 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dyvy'] = np.array(
        [2 * sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dzvy'] = np.array(
        [2 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dxvz'] = np.array(
        [3 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dyvz'] = np.array(
        [3 * sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['dzvz'] = np.array(
        [3 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['delv'] = np.array(
        [cosx[i] * sinx[j] * sinx[k] + 2 * sinx[i] * cosx[j] * sinx[k] + 3 * sinx[i] * sinx[j] * cosx[k] for i in
         range(npoint) for j in range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    return filename, dic_quant, dic_param


class TestRecordFromV:
    def test_record_v(self, init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            record_v(f, init_arg_fv[1], init_arg_fv[2])
        with h5.File(init_arg_fv[0], 'r') as f:
            for quant in ['vx', 'vy', 'vz']:
                assert np.max(np.abs(init_arg_fv[1][quant] - np.array(f[quant]))) < init_arg_fv[2][
                    'precision'], f"error on {quant} recording"

    def test_record_w(self, init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            record_w(f, init_arg_fv[1], init_arg_fv[2])
        with h5.File(init_arg_fv[0], 'r') as f:
            for quant in ['wx', 'wy', 'wz']:
                assert np.max(np.abs(init_arg_fv[1][quant] - np.array(f[quant]))) < init_arg_fv[2][
                    'precision'], f"error on {quant} recording"

    def test_record_gradv(self, init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            record_gradv(f, init_arg_fv[1], init_arg_fv[2])
        with h5.File(init_arg_fv[0], 'r') as f:
            for quant in ['dxvx', 'dyvx', 'dzvx', 'dxvy', 'dyvy', 'dzvy', 'dxvz', 'dyvz', 'dzvz']:
                assert np.max(np.abs(init_arg_fv[1][quant] - np.array(f[quant]))) < init_arg_fv[2][
                    'precision'], f"error on {quant} recording"

    def test_record_delv(self, init_arg_fv):
        with h5.File(init_arg_fv[0], 'w') as f:
            record_delv(f, init_arg_fv[1], init_arg_fv[2])
        with h5.File(init_arg_fv[0], 'r') as f:
            for quant in ['delv', ]:
                assert np.max(np.abs(init_arg_fv[1][quant] - np.array(f[quant]))) < init_arg_fv[2][
                    'precision'], f"error on {quant} recording"


@pytest.fixture
def init_arg_frho():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['rho'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['drhox'] = np.array(
        [cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['drhoy'] = np.array(
        [sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['drhoz'] = np.array(
        [sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    return filename, dic_quant, dic_param


class TestRecordFromRho:
    def test_record_rho(self, init_arg_frho):
        with h5.File(init_arg_frho[0], 'w') as f:
            record_rho(f, init_arg_frho[1], init_arg_frho[2])
        with h5.File(init_arg_frho[0], 'r') as f:
            for quant in ['rho', ]:
                assert np.max(np.abs(init_arg_frho[1][quant] - np.array(f[quant]))) < init_arg_frho[2][
                    'precision'], f"error on {quant} recording"

    def test_record_delrho(self, init_arg_frho):
        with h5.File(init_arg_frho[0], 'w') as f:
            record_delrho(f, init_arg_frho[1], init_arg_frho[2])
        with h5.File(init_arg_frho[0], 'r') as f:
            for quant in ['drhox', 'drhoy', 'drhoz']:
                assert np.max(np.abs(init_arg_frho[1][quant] - np.array(f[quant]))) < init_arg_frho[2][
                    'precision'], f"error on {quant} recording"


@pytest.fixture
def init_arg_fp():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['rho'] = 2 * np.ones([npoint, npoint, npoint])
    dic_quant['ppar'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['pperp'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect = {}
    dic_expect['Ippar'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Ipperp'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['ppar'] = np.array(
        [1 / 2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['pperp'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['ugyr'] = np.array(
        [5 / 4 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['piso'] = np.array(
        [5 / 6 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['uiso'] = np.array(
        [5 / 4 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['duisox'] = np.array(
        [5 / 4 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['duisoy'] = np.array(
        [5 / 4 * sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['duisoz'] = np.array(
        [5 / 4 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    return filename, dic_quant, dic_param, dic_expect


class TestRecordFromP:
    def test_record_Igyrp(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            record_Igyrp(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['Ippar', 'Ipperp']:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"

    def test_record_gyrp(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            record_gyrp(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['ppar', 'pperp']:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"

    def test_record_gyru(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            record_gyru(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['ugyr', ]:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"

    def test_record_isop(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            record_isop(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['piso', ]:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"

    def test_record_isou(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            record_isou(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['uiso', ]:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"

    def test_record_delisou(self, init_arg_fp):
        with h5.File(init_arg_fp[0], 'w') as f:
            record_delisou(f, init_arg_fp[1], init_arg_fp[2])
        with h5.File(init_arg_fp[0], 'r') as f:
            for quant in ['duisox', 'duisoy', 'duisoz']:
                assert np.max(np.abs(init_arg_fp[3][quant] - np.array(f[quant]))) < init_arg_fp[2][
                    'precision'], f"error on {quant} recording"


@pytest.fixture
def init_arg_fb():
    filename = "test_file.h5"
    npoint = 50
    x = np.arange(0, npoint) / npoint * 2 * np.pi
    sinx = np.sin(x)
    cosx = np.cos(x)
    dic_quant = {}
    dic_quant['rho'] = 4 * np.ones([npoint, npoint, npoint])
    dic_quant['bx'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['by'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_quant['bz'] = np.array(
        [3 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect = {}
    dic_expect['Ibx'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Iby'] = np.array(
        [2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Ibz'] = np.array(
        [3 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['Ipm'] = np.array(
        [7 * (sinx[i] * sinx[j] * sinx[k]) * (sinx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['bx'] = np.array(
        [1 / 2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['by'] = np.array(
        [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_expect['bz'] = np.array(
        [3 / 2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
         range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['pm'] = np.array(
        [7 / 4 * (sinx[i] * sinx[j] * sinx[k]) * (sinx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['delb'] = np.array(
        [1 / 2 * cosx[i] * sinx[j] * sinx[k] + sinx[i] * cosx[j] * sinx[k] + 3 / 2 * sinx[i] * sinx[j] * cosx[k] for i
         in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['Ijx'] = np.array(
        [(3 * sinx[i] * cosx[j] * sinx[k] - 2 * sinx[i] * sinx[j] * cosx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['Ijy'] = np.array(
        [(sinx[i] * sinx[j] * cosx[k] - 3 * cosx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['Ijz'] = np.array(
        [(2 * cosx[i] * sinx[j] * sinx[k] - sinx[i] * cosx[j] * sinx[k]) for i in range(npoint) for j in range(npoint)
         for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['jx'] = np.array(
        [1 / 4 * (3 * sinx[i] * cosx[j] * sinx[k] - 2 * sinx[i] * sinx[j] * cosx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['jy'] = np.array(
        [1 / 4 * (sinx[i] * sinx[j] * cosx[k] - 3 * cosx[i] * sinx[j] * sinx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['jz'] = np.array(
        [1 / 4 * (2 * cosx[i] * sinx[j] * sinx[k] - sinx[i] * cosx[j] * sinx[k]) for i in range(npoint) for j in
         range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
    dic_expect['delj'] = np.array([1 / 4 * ((3 * cosx[i] * cosx[j] * sinx[k] - 2 * cosx[i] * sinx[j] * cosx[k]) + (
            sinx[i] * cosx[j] * cosx[k] - 3 * cosx[i] * cosx[j] * sinx[k]) + (
                                                    2 * cosx[i] * sinx[j] * cosx[k] - sinx[i] * cosx[j] * cosx[k]))
                                   for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
        (npoint, npoint, npoint))
    dic_param = {'N': [npoint, npoint, npoint], 'c': [x[1], x[1], x[1]], 'precision': x[1] * x[1] * x[1] * x[1]}
    yield filename, dic_quant, dic_param, dic_expect
    os.remove(filename)


class TestRecordFromB:
    def test_record_Ij(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_Ij(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Ijx', 'Ijy', 'Ijz']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_j(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_j(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['jx', 'jy', 'jz']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_delj(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_delj(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['delj', ]:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_Ib(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_Ib(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Ibx', 'Iby', 'Ibz']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_Ipm(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_Ipm(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['Ipm', ]:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_b(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_b(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['bx', 'by', 'bz']:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_delb(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_delb(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['delb', ]:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"

    def test_record_pm(self, init_arg_fb):
        with h5.File(init_arg_fb[0], 'w') as f:
            record_pm(f, init_arg_fb[1], init_arg_fb[2])
        with h5.File(init_arg_fb[0], 'r') as f:
            for quant in ['pm', ]:
                assert np.max(np.abs(init_arg_fb[3][quant] - np.array(f[quant]))) < init_arg_fb[2][
                    'precision'], f"error on {quant} recording"


# A FAIRE DE MEME POUR LES AUTRES FONCTIONS RECORD (bj)
class TestDataProcessOCA:
    """voir contenu du fichier output_process_[...].txt pour vérifier le détail des étapes et le contenu du fichier .h5 de sortie."""
    pass


class TestDataBinning:
    """voir contenu du fichier output_process_[...].txt."""
    pass


class TestCheck:
    """voir contenu du fichier output_process_[...].txt."""
    pass


class TestBinAnArray:
    def test_bin2_ones(self):
        tab = np.ones((10, 10, 10))
        bin = 2
        result = bin_an_array(tab, 2)
        expected_result = np.ones((5, 5, 5))
        assert np.array_equal(result, expected_result), f"error on the binning of an array"

    def test_bin2_gradx(self):
        x = np.arange(0, 10)
        tab = np.ones((10, 10, 10))
        for i in range(10):
            tab[i] = tab[i] * x[i]
        bin = 2
        result = bin_an_array(tab, 2)
        print(tab[:, 0, 0])
        print(result[:, 0, 0])
        expected_result = np.ones((5, 5, 5))
        expected_x = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
        for i in range(5):
            expected_result[i] = expected_result[i] * expected_x[i]
        assert np.array_equal(result, expected_result), f"error on the binning of an array"


class TestBinArraysInh5:
    """voir contenu du fichier output_process_[...].txt."""
    pass


class TestInputfileToDict:
    """voir contenu du fichier output_process_[...].txt."""
    pass
