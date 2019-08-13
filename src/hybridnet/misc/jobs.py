import yaml
import fnmatch
import re
import os
import sys
import logging
import time
import random
from ..misc.tools import DotDict, input_with_timeout
from ..misc.config import merge
from ..misc import config as _config
config = _config.get()


def preprocess_jobs_grid(jobs, id, job_options):

    # parse grid ({key: [vals...]})
    if job_options.job.grid:
        # select key of the first grid
        key = list(job_options.job.grid.keys())[0]

        # iterate on the grid
        for val in job_options.job.grid[key]:
            # init params of the sub-job with this key=val setting
            sub_job_options = DotDict(job_options.get_dict())
            del sub_job_options.job.grid[key]
            if not sub_job_options.job.grid:
                del sub_job_options.job.grid

            # set key=val
            key_path = key.split('.')
            sub_opt = sub_job_options
            for key_i in key_path[:-1]:
                if not key_i in sub_opt:
                    sub_opt[key_i] = DotDict({})
                sub_opt = sub_opt[key_i]

            sub_opt[key_path[-1]] = val

            # send the sub-job back in (will be completed or added)
            preprocess_jobs_grid(jobs, id, sub_job_options)

    elif job_options.job.grid_list:

        for additional_options in job_options.job.grid_list:

            # init params of the sub-job
            sub_job_options = DotDict(job_options.get_dict())
            del sub_job_options.job.grid_list

            # set key=val
            for key, val in additional_options.items():
                key_path = key.split('.')
                sub_opt = sub_job_options
                for key_i in key_path[:-1]:
                    if not key_i in sub_opt:
                        sub_opt[key_i] = DotDict({})
                    sub_opt = sub_opt[key_i]

                sub_opt[key_path[-1]] = val

            # send the sub-job back in (will be completed or added)
            preprocess_jobs_grid(jobs, id, sub_job_options)

    # we're done with this, add it
    else:
        sub_id = id.format(**job_options)
        if job_options.path:
            job_options.path = job_options.path.format(**job_options)
        jobs[sub_id] = job_options

    return jobs


def preprocess_jobs(jobs):
    jobs_ids = list(jobs.keys())
    for id in jobs_ids:
        if jobs[id].disabled:
            print("INFO: Remove disabled job %s" % id)
            del jobs[id]
        elif jobs[id].job.grid or jobs[id].job.grid_list:
            print("INFO: Expanding job %s" % id)
            jobs = preprocess_jobs_grid(jobs, id, jobs[id])
            del jobs[id]
    return jobs


def run(id, job_options, additional_options={}):
    print("\n\n"
          "==================================================\n"
          " >> RUNNING JOB: %s\n"
          " > Options:\n%s" % (id, job_options))

    # init
    job_options = DotDict(merge(job_options, additional_options))
    id = id.format(**job_options)
    if job_options.path: job_options.path = job_options.path.format(**job_options)
    if job_options.path_load: job_options.path_load = job_options.path_load.format(**job_options)
    job_options.id = id

    # job disabled
    if job_options.disabled:
        print('INFO: Job %s is disabled' % id)
        return

    # job path
    if not job_options.path:
        job_options.path = id + "/"
    if not job_options.path.startswith(config.experiments.results_path) and not job_options.path.startswith("/"):
        job_options.path = config.experiments.results_path + job_options.path

    # job already ran?
    ran = False
    if job_options.path.endswith("/"):
        if os.path.exists(job_options.path):
            ran = True
        else:
            os.makedirs(job_options.path)
    else:
        dir, pattern = os.path.split(job_options.path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif len(fnmatch.filter(os.listdir(dir), pattern + "*")) > 0:
            ran = True
    if ran and not job_options.force:
        print('INFO: Job %s already ran. Add "force: True" option to force rerun or use --force' % id)
        return

    # remove previous tensorboard
    if os.path.exists(config.experiments.tensorboard_path+id):
        i = ""
        while os.path.exists(config.experiments.tensorboard_path+id+"_old"+str(i)):
            i = 2 if i == "" else i + 1
        os.rename(config.experiments.tensorboard_path+id, config.experiments.tensorboard_path+id+"_old"+str(i))

    # Logger init
    logger = logging.getLogger(id)
    logger.setLevel(logging.DEBUG)
    log_level_console = job_options.log_level if job_options.log_level is not None else logging.INFO
    log_level_file = job_options.log_level_file if job_options.log_level_file is not None else logging.DEBUG
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)', datefmt='%Y-%m-%d %H:%M:%S')
    logger_console = logging.StreamHandler(sys.stdout)
    logger_console.setLevel(log_level_console)
    logger_console.setFormatter(log_format)
    logger.addHandler(logger_console)
    logger_file = logging.FileHandler(job_options.path+"job.log")
    logger_file.setLevel(log_level_file)
    logger_file.setFormatter(log_format)
    logger.addHandler(logger_file)

    # Tensorflow init
    if not job_options.no_tf:
        import tensorflow as tf
        from tensorflow.python.keras import backend as K
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=tf_config))

    # load job module
    if not job_options.job.get("class"):
        print('ERROR: No class given for job %s' % id)
        return

    try:
        from_list = [job_options.job.get("class")]
        if not job_options.job.absolute:
            if not job_options.job.module:
                module = __import__("models", globals(), locals(), from_list, 2)
            else:
                module = __import__("models." + job_options.job.module, globals(), locals(), from_list, 2)
        else:
            module = __import__(job_options.job.module, globals(), locals(), from_list, 0)

        # run job
        el = getattr(module, job_options.job.get("class"))(job_options)
        if not job_options.job.run:
            job_options.job.run = ["run"]
        if not isinstance(job_options.job.run, list):
            job_options.job.run = [job_options.job.run]
        for method_name in job_options.job.run:
            getattr(el, method_name)()
    except KeyboardInterrupt:
        logger.error("Job interrupted by CTRL+C")
        time.sleep(1)
        stop_input = input("Continue to next job? [n/no to stop]")
        if stop_input == "n" or stop_input == "no":
            exit()
    except:
        logger.error("Error during job", exc_info=True)

        # input with 60s timeout
        time.sleep(1)
        stop_input = input_with_timeout("Continue to next job? [n/no to stop]")
        if stop_input == "n" or stop_input == "no":
            exit()

    if not job_options.no_tf:
        K.clear_session()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Jobs runner.')
    parser.add_argument('--job-file', help='YAML file containing list of jobs that should be ran', type=str,
                        default=config.experiments.default_file)
    parser.add_argument('--job', help='Specific job to run', type=str)
    parser.add_argument('--job-regex', help='Filter jobs to run with regex', type=str)
    parser.add_argument('--job-filter', help='Filter jobs to run with simple wildcard filter', type=str)
    parser.add_argument('--force', help='Force running jobs', action='store_true')
    parser.add_argument('--no-tf', help='No Tensorflow', action='store_true')

    args = parser.parse_args()

    with open(args.job_file, "r") as f:
        jobs = DotDict(yaml.load(f))
    if "config" in jobs:
        _config.set_custom(jobs.pop("config"))
        config = _config.get()
        print("INFO: Custom config found and processed")
    print(_config.get())

    print("INFO: Jobs found: %s" % list(jobs.keys()))

    # Additional jobs
    additional_options = {}
    if args.force:
        additional_options["force"] = True
    if args.no_tf:
        additional_options["no_tf"] = True

    # Preprocess
    jobs = preprocess_jobs(jobs)
    print("INFO: Jobs after pre-processing: %s" % list(jobs.keys()))

    #jobfile_name, jobfile_ext = os.path.splitext(args.job_file)
    #with open("{name}_ran{ext}".format(name=jobfile_name, ext=jobfile_ext), "w") as f:
    #    yaml.dump(jobs.get_dict(), f, default_flow_style=False)

    # Run
    if args.job is not None:
        if args.job in jobs:
            print("INFO: Soon running job: %s" % args.job)
            time.sleep(1.5 + random.random())
            run(args.job, jobs[args.job], additional_options)
        else:
            print("ERROR: " + args.job + " not found")
    else:
        if args.job_filter is not None:
            jobs = {k: v for (k, v) in jobs.items() if fnmatch.fnmatch(k, args.job_filter)}
            print("INFO: Jobs to run after wildcard filter: %s" % list(jobs.keys()))
        if args.job_regex is not None:
            jobs = {k: v for (k, v) in jobs.items() if re.match(args.job_regex, k)}
            print("INFO: Jobs to run after regex filter: %s" % list(jobs.keys()))

        print("INFO: Soon running jobs...")

        if len(jobs) == 0:
            print("ERROR: No job to run")
        for job in sorted(jobs):
            print("INFO: Starting job %s" % job)
            run(job, jobs[job], additional_options)
