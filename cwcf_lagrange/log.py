import numpy as np
import pandas as pd
import time, os, warnings

import sklearn.metrics as metrics

from config import config

from agent import PerfAgent
from env import SeqEnvironment
import utils

#==============================
class Log:
    def __init__(self, data, hpc_p, costs, brain, groups=None, log_name=None):
        self.data = data
        self.hpc_p = hpc_p
        self.costs = costs
        self.groups = groups
        self.brain = brain

        self.LOG_TRACKED_STATES = np.array(config.LOG_TRACKED_STATES, dtype=np.float32)
        self.LEN = len(self.LOG_TRACKED_STATES)

        self.log_name = log_name
        self.log_to_file = log_name is not None

        if self.log_to_file:
            if config.BLANK_INIT:
                mode = "w"
            else:
                mode = "a"

            self.files = []
            for i in range(self.LEN):
                self.files.append( open("run_{}_{}.dat".format(log_name, i), mode) )

            self.perf_file = open("run_{}_perf.dat".format(log_name), mode)

        self.time = 0

    def log_q(self):
        if self.log_to_file:
            q = self.brain.predict_np(self.LOG_TRACKED_STATES)

            for i in range(self.LEN):
                w = q[i].data

                for k in w:
                    self.files[i].write('%.4f ' % k)

                self.files[i].write('\n')
                self.files[i].flush()

    def print_speed(self):
        if self.time == 0:
            self.time = time.perf_counter()
            return

        now = time.perf_counter()
        elapsed = now - self.time
        self.time = now

        samples_processed = config.LOG_EPOCHS * config.EPOCH_STEPS * config.AGENTS
        updates_processed = config.LOG_EPOCHS
        updates_total = config.LOG_EPOCHS * config.BATCH_SIZE

        fps_smpl = samples_processed / elapsed
        fps_updt = updates_processed / elapsed
        fps_updt_t = updates_total / elapsed

        print("Perf.: {:.0f} gen_smp/s, {:.1f} upd/s, {:.1f} upd_steps/s".format(fps_smpl, fps_updt, fps_updt_t))

    def eval_avg_cost(self):
        env = SeqEnvironment(self.data, self.hpc_p, self.costs, self.groups)
        agent = PerfAgent(env, self.brain)

        _fc   = 0.

        while True:
            s, a, r, s_, done, info = agent.step()

            if np.all(done == -1):
                break

            finished   = (done ==  1)       # episode finished
            terminated = (done == -1)       # no more data

            _fc   += np.sum(info['fc']) + np.sum(info['hpc_fc'])

        data_len = len(self.data)
        _fc    /= data_len

        return _fc

    def log_perf(self, histogram=False, verbose=False, save_probs=False):
        env = SeqEnvironment(self.data, self.hpc_p, self.costs)
        agent = PerfAgent(env, self.brain)

        # compute metrics
        _r    = 0.
        _fc   = 0.
        _len  = 0.
        _corr = 0.
        _hpc  = 0.
        _lens = []
        _lens_hpc = []

        y_vals = []
        correct = []
        done_bool = []
        qs = []
        pred_y = []
        true_y = []

        # f_cnt = 0
        while True:
            # import pdb; pdb.set_trace()
            # utils.print_progress(np.sum(self.done), self.agents, step=1)
            y_vals.append(agent.env.y.copy())
            try: s, a, r, s_, w, done, info = agent.step()
            except IndexError: warnings.warn(f'dir: {os.getcwd()}, lambda: {config.FEATURE_FACTOR}'); time.sleep(10000000)
            correct.append(info['corr'].copy())
            done_bool.append(done.copy())
            qs.append(agent.brain.predict_np(s))

            if np.all(done == -1):
                break

            finished   = (done ==  1)		# episode finished
            terminated = (done == -1)		# no more data

            # f_cnt += np.sum(finished)
            # utils.print_progress(f_cnt, len(self.data))

            # rescale feature costs with lambda
            r[~finished] *= config.FEATURE_FACTOR 

            _r    += np.sum( (r * w)[~terminated] ) # TODO
            _fc   += np.sum( (info['fc']) + np.sum(info['hpc_fc']) * w )
            _corr += np.sum( info['corr'] * w )
            _len  += np.sum((~terminated) * w )
            _hpc  += np.sum( info['hpc'] * w )

            pred_y.extend( filter(lambda x: x is not None, info['pred_y']) )
            true_y.extend( filter(lambda x: x is not None, info['true_y']) )

            if histogram:
                finished_hpc   = finished * info['hpc']
                finished_nohpc = finished * ~info['hpc']

                _lens.append(info['eplen'][finished_nohpc])
                _lens_hpc.append(info['eplen'][finished_hpc])
        results = pd.DataFrame(np.vstack((
            np.hstack(y_vals),
            np.hstack(correct),np.hstack(done_bool))).T,columns=['y','correct','done'])
        if save_probs:
            results.to_csv('run_{}_allpreds.csv'.format(self.log_name))
            np.savetxt('run_{}_allqs.csv'.format(self.log_name),np.vstack(qs),delimiter=',')


        data_len = len(self.data)
        _r    /= data_len
        _fc   /= data_len
        _corr /= data_len
        _len  /= data_len
        _hpc  /= data_len

        print("{} R: {:.5f} | L: {:.5f} | FC: {:.5f} | HPC: {:.5f} | C: {:.5f}".format(self.log_name, _r, _len, _fc, _hpc, _corr))

        if self.log_to_file:
            print("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(_r, _len, _fc, _hpc, _corr), file=self.perf_file, flush=True)

        if histogram:
            _lens = np.concatenate(_lens).flatten()
            _lens_hpc = np.concatenate(_lens_hpc).flatten()

            # print("Writing histogram...")
            with open('run_{}_histogram.dat'.format(self.log_name), 'w') as file:
                for x in _lens:
                    file.write("{} ".format(x))

            with open('run_{}_histogram_hpc.dat'.format(self.log_name), 'w') as file:
                for x in _lens_hpc:
                    file.write("{} ".format(x))

        if verbose:
            conf_matrix = metrics.confusion_matrix(true_y, pred_y)
            print("Confusion matrix:")
            print(conf_matrix)

            print("Classification report:")
            print(metrics.classification_report(true_y, pred_y))

            # compute balanced accuracy
            cls_acc = np.diag(conf_matrix) / np.sum(conf_matrix, 1) # accuracy per class
            cls_avg = np.sum(cls_acc) / conf_matrix.shape[0]
            print("Rebalanced accuracy: {}".format(cls_avg))

        return _r, _len, _fc, _hpc, _corr
