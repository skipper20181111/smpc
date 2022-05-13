import numpy as np

from baseCommon.plogger import *
from federatedml.woe.woe_iv_ca_square_basic import *
# from secureprotol.encrypt import PaillierEncrypt
from baseCommon.logger import LogClass, ON
import pandas as pd
import multiprocessing as mp

# logclass=LogClass("testlinrlog.txt")
# LOGGER = logclass.get_Logger("levelname!!", ON.DEBUG)


LoggerFactory.set_directory(directory="./")
LOGGER = getLogger()


# interdata = [1,2]#根据求交结果导入
# data = interdata


class woe_iv_ca_square_host():

    def __init__(self):

        self.uuid = None
        self.alldata = None  # linear_host.public_key = None
        self.data = None
        #
        self.hostoutfile = None
        self.testdata = None
        self.jobid = None
        self.name = None
        # self.hostcsv=None
        self.filepath = None
        #
        self.w = None
        self.save_woe_translate = {}

    def load_data(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "load_data":
            (encrypt_y, self.jobid, self.name, self.filepath, self.hostoutfile) = reqdata
            orHostDataName = inter_dataname_TANS(self.filepath)
            # filePath = get_path_by_name(orHostDataName)
            filePath = get_path_from_db(orHostDataName)
            self.filepath = concat_filepath(self.filepath, self.jobid, filePath)

            filename = orHostDataName + "_inter_woe.csv"
            self.save_woe_path = concat_filepath(filename, self.jobid, filePath)

            self.data = pd.read_csv(self.filepath)
            self.data = insert_encrypt_y(self.data, encrypt_y)
            return 0, 'succes'
        else:
            LOGGER.info("discrete_combination Error!")
            return 500, "discrete_combination Error!"

    def discrete_feacher(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "discrete_feacher":
            feacher = reqdata[0]
            self.data, feacher_category = discrete_encode(self.data, feacher)
            feacher_category_index = list(set(self.data[feacher]))
            y_list = discrete_ylist(self.data, feacher, feacher_category_index)

            return 0, [feacher_category_index, y_list, feacher_category]
        else:
            LOGGER.info("discrete_feacher Error!")
            return 500, "discrete_feacher Error!"

    def discrete_combination(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "discrete_combination":
            feacher, dictionary, feacher_category = reqdata
            m, n = self.data.shape
            for i in range(m):
                for comb in dictionary:
                    a = int(comb)
                    b = int(self.data.loc[i, feacher])
                    if feacher_category[b] in dictionary[a]:
                        self.data.loc[i, feacher] = comb
            print(self.data, set(self.data.loc[:, feacher]))
            self.save_woe_translate[feacher] = [2, [feacher_category, dictionary]]
            return 0, 'success'
        else:
            LOGGER.info("discrete_combination Error!")
            return 500, "discrete_combination Error!"

    def host_woe_transform_get_y_list(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "host_woe_transform_get_y_list":
            feacher = reqdata
            feacher_category = list(set(self.data[feacher]))
            y_list = discrete_ylist(self.data, feacher, feacher_category)
            return 0, [y_list, feacher_category]
        else:
            LOGGER.info("host_woe_transform_get_y_list Error!")
            return 500, "host_woe_transform_get_y_list Error!"

    def host_woe_transform_give_woe(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "host_woe_transform_give_woe":
            feacher, woelist, iv, feacher_category = reqdata
            self.data = woe_transform(feacher_category, woelist, self.data, feacher)
            self.save_woe_translate[feacher].append([feacher_category, woelist, iv])
            print(self.save_woe_translate)
            print(self.data, '这是host %s 的最终结果' % feacher)
            self.data.to_csv(self.save_woe_path, index=0)
            return 0, iv
        else:
            LOGGER.info("host_woe_transform_give_woe Error!")
            return 500, "host_woe_transform_give_woe Error!"

    def continuous_feacher(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "continuous_feacher":
            method, feacher, bin_num_threshold = reqdata
            feacher_list, y_list = give_bin_ypoint_and_ylist(method, self.data, feacher,
                                                             bin_num_threshold * 20)
            return 0, [feacher_list, y_list]
        else:
            LOGGER.info("continuous_feacher Error!")
            return 500, "continuous_feacher Error!"

    def continue_combination(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "continue_combination":
            feacher, feacher_cut_point_list = reqdata
            self.save_woe_translate[feacher] = [1, feacher_cut_point_list]
            self.data = continuation_combination(feacher, self.data, feacher_cut_point_list)
            print(self.data, set(self.data.loc[:, feacher]))
            return 0, 'success'
        else:
            LOGGER.info("continue_combination Error!")
            return 500, "continue_combination Error!"
