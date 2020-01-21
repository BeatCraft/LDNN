# coding: utf-8

import json
import struct

CMD_TYPE_UNKNOWN = 0
CMD_TYPE_CHANGE_WEIGHT = 1
CMD_TYPE_EVAL_WEIGHT = 2
CMD_TYPE_NODE_QUIT = 9

CMD_FMT_UNKNOWN = 0
CMD_FMT_BIN = 1
CMD_FMT_JSON = 2

RES_OK = 0
RES_ERROR = 1

class Command(object):
    def __init__(self):
        self.cmd_type = CMD_TYPE_UNKNOWN
        self.name = "Command"

    def __str__(self):
        return "%s(%d)" % (self.name, self.cmd_type)

    def encode_json(self):
        raise NotImplementedError

    def encode_bin(self):
        raise NotImplementedError

class Result(Command):
    def __init__(self, result_code):
        super(Result, self).__init__()
        self.name = "Result"
        self.result_code = result_code

    def __str__(self):
        return super(Result, self).__str__() \
               + " code " + str(self.result_code)

    def encode_json(self):
        return json.dumps({
            "cmd_type": self.cmd_type,
            "result_code": self.result_code,
        })

    def decode_json(self, obj):
        self.result_code = obj["result_code"]

    def encode_bin(self):
        data = struct.pack(">II", self.cmd_type, self.result_code)
        return data

    def decode_bin(self, data):
        _type, code = struct.unpack(">II", data[:8])
        self.result_code = code
# Quit
class QuitCommand(Command):
    def __init__(self):
        super(QuitCommand, self).__init__()
        self.cmd_type = CMD_TYPE_NODE_QUIT
        self.name = "QuitCommand"

    def encode_bin(self):
        data = struct.pack(">I", self.cmd_type)
        return data

    def decode_bin(self, data):
		return None

# ChangeWeight

class Weight(object):
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __repr__(self):
        return "(%d, %f)" % (self.index, self.value)

class ChangeWeightCommand(Command):
    def __init__(self):
        super(ChangeWeightCommand, self).__init__()
        self.cmd_type = CMD_TYPE_CHANGE_WEIGHT
        self.name = "ChangeWeightCommand"
        self.weight_list = []

    def __str__(self):
        return super(ChangeWeightCommand, self).__str__() \
               + " weights " + str(len(self.weight_list)) \
               + " " + str(self.weight_list)

    def add_weight(self, index, value):
        self.weight_list.append(Weight(index, value))

    def encode_json(self):
        weight_objs = []
        for weight in self.weight_list:
            weight_objs.append({"index": weight.index, "value": weight.value})
        return json.dumps({
            "cmd_type": self.cmd_type,
            "weight_list": weight_objs,
        })

    def decode_json(self, obj):
        weight_list = []
        for item in obj["weight_list"]:
            weight_list.append(Weight(item["index"], item["value"]))
        self.weight_list = weight_list

    def encode_bin(self):
        data = struct.pack(">II", self.cmd_type, len(self.weight_list))
        for weight in self.weight_list:
            data += struct.pack(">If", weight.index, weight.value)
        return data

    def decode_bin(self, data):
        _type, _len = struct.unpack(">II", data[:8])
        self.weight_list = []
        for i in xrange(_len):
            index, value = struct.unpack(">If", data[8+i*8:16+i*8])
            self.weight_list.append(Weight(index, value))

class ChangeWeightResult(Result):
    def __init__(self, result_code):
        super(ChangeWeightResult, self).__init__(result_code)
        self.cmd_type = CMD_TYPE_CHANGE_WEIGHT
        self.name = "ChangeWeightResult"

# EvalWeight

class EvalWeightTask(object):
    def __init__(self, task_id, weight_index, weight_value):
        self.task_id = task_id
        self.weight_index = weight_index
        self.weight_value = weight_value

    def __repr__(self):
        return "(%d, %d, %f)" % (self.task_id, self.weight_index, self.weight_value)

class EvalWeightCommand(Command):
    def __init__(self):
        super(EvalWeightCommand, self).__init__()
        self.cmd_type = CMD_TYPE_EVAL_WEIGHT
        self.name = "EvalWeightCommand"
        self.task_list = []

    def __str__(self):
        return super(EvalWeightCommand, self).__str__() \
               + " tasks " + str(len(self.task_list)) \
               + " " + str(self.task_list)

    def add_task(self, task_id, weight_index, weight_value):
        self.task_list.append(EvalWeightTask(task_id, weight_index, weight_value))

    def encode_json(self):
        task_objs = []
        for task in self.task_list:
            task_objs.append({
                "task_id": task.task_id,
                "weight_index": task.weight_index,
                "weight_value": task.weight_value,
            })
        return json.dumps({
            "cmd_type": self.cmd_type,
            "task_list": task_objs,
        })

    def decode_json(self, obj):
        task_list = []
        for item in obj["task_list"]:
            task = EvalWeightTask(item["task_id"],
                                item["weight_index"],
                                item["weight_value"])
            task_list.append(task)
        self.task_list = task_list

    def encode_bin(self):
        data = struct.pack(">II", self.cmd_type, len(self.task_list))
        for task in self.task_list:
            data += struct.pack(">IIf", task.task_id, task.weight_index, task.weight_value)
        return data

    def decode_bin(self, data):
        _type, _len = struct.unpack(">II", data[:8])
        self.task_list = []
        for i in xrange(_len):
            task_id, weight_index, weight_value = struct.unpack(">IIf", data[8+i*12:20+i*12])
            self.task_list.append(EvalWeightTask(task_id, weight_index, weight_value))

class WeightEval(object):
    def __init__(self, task_id, eval_value):
        self.task_id = task_id
        self.eval_value = eval_value

    def __repr__(self):
        return "(%d, %f)" % (self.task_id, self.eval_value)

class EvalWeightResult(Result):
    def __init__(self, result_code):
        super(EvalWeightResult, self).__init__(result_code)
        self.cmd_type = CMD_TYPE_EVAL_WEIGHT
        self.name = "EvalWeightResult"
        self.eval_list = []

    def __str__(self):
        return super(EvalWeightResult, self).__str__() \
               + " evals " + str(len(self.eval_list)) \
               + " " + str(self.eval_list)

    def add_eval(self, task_id, eval_value):
        self.eval_list.append(WeightEval(task_id, eval_value))

    def encode_json(self):
        eval_objs = []
        for _eval in self.eval_list:
            eval_objs.append({
                "task_id": _eval.task_id,
                "eval_value": _eval.eval_value,
            })
        return json.dumps({
            "cmd_type": self.cmd_type,
            "result_code": self.result_code,
            "eval_list": eval_objs,
        })

    def decode_json(self, obj):
        super(EvalWeightResult, self).decode_json(obj)
        eval_list = []
        for item in obj["eval_list"]:
            _eval = WeightEval(item["task_id"], item["eval_value"])
            eval_list.append(_eval)
        self.eval_list = eval_list

    def encode_bin(self):
        data = super(EvalWeightResult, self).encode_bin()
        data += struct.pack(">I", len(self.eval_list))
        for _eval in self.eval_list:
            data += struct.pack(">If", _eval.task_id, _eval.eval_value)
        return data

    def decode_bin(self, data):
        super(EvalWeightResult, self).decode_bin(data)
        eval_len = struct.unpack(">I", data[8:12])[0]
        eval_list = []
        for i in xrange(eval_len):
            task_id, eval_value = struct.unpack(">If", data[12+i*8:20+i*8])
            eval_list.append(WeightEval(task_id, eval_value))
        self.eval_list = eval_list

# decode

cmd_class_map = {
    CMD_TYPE_CHANGE_WEIGHT: ChangeWeightCommand,
    CMD_TYPE_EVAL_WEIGHT: EvalWeightCommand,
    CMD_TYPE_NODE_QUIT: QuitCommand,
}

def decode_command(data, fmt=CMD_FMT_UNKNOWN):
    if fmt == CMD_FMT_UNKNOWN:
        if data[0] == "{":
            fmt = CMD_FMT_JSON
        else:
            fmt = CMD_FMT_BIN

    if fmt == CMD_FMT_JSON:
        obj = json.loads(data)
        cmd = cmd_class_map[obj["cmd_type"]]()
        cmd.decode_json(obj)
        return cmd
    else:
        _type = struct.unpack(">I", data[:4])[0]
        cmd = cmd_class_map[_type]()
        cmd.decode_bin(data)
        return cmd

res_class_map = {
    CMD_TYPE_CHANGE_WEIGHT: ChangeWeightResult,
    CMD_TYPE_EVAL_WEIGHT: EvalWeightResult,
}

def decode_result(data, fmt=CMD_FMT_UNKNOWN):
    if fmt == CMD_FMT_UNKNOWN:
        if data[0] == "{":
            fmt = CMD_FMT_JSON
        else:
            fmt = CMD_FMT_BIN

    if fmt == CMD_FMT_JSON:
        obj = json.loads(data)
        res = res_class_map[obj["cmd_type"]](obj["result_code"])
        res.decode_json(obj)
        return res
    else:
        _type, code = struct.unpack(">II", data[:8])
        res = res_class_map[_type](code)
        res.decode_bin(data)
        return res

# test

def print_bin_cmd(data):
    _type, _len = struct.unpack(">II", data[:8])
    print("%d %d" % (_type, _len))
    for i in xrange(_len):
        if _type == CMD_TYPE_CHANGE_WEIGHT:
            index, value = struct.unpack(">If", data[8+i*8:16+i*8])
            print(" %d %f" % (index, value))
        elif _type == CMD_TYPE_EVAL_WEIGHT:
            _id, index, value = struct.unpack(">IIf", data[8+i*12:20+i*12])
            print(" %d %d %f" % (_id, index, value))

def print_bin_res(data):
    _type, code = struct.unpack(">II", data[:8])
    print("%d %d" % (_type, code))
    if _type == CMD_TYPE_EVAL_WEIGHT:
        _len = struct.unpack(">I", data[8:12])[0]
        for i in xrange(_len):
            _id, value = struct.unpack(">If", data[12+i*8:20+i*8])
            print(" %d %f" % (_id, value))

def test_cmd(cmd):
    print(cmd)
    print("> encode json")
    payload = cmd.encode_json()
    print(payload)
    print("> decode json")
    cmd = decode_command(payload)
    print(cmd)
    print("> encode bin")
    payload = cmd.encode_bin()
    print_bin_cmd(payload)
    print("> decode bin")
    cmd = decode_command(payload)
    print(cmd)

def test_res(cmd):
    print(cmd)
    print("> encode json")
    payload = cmd.encode_json()
    print(payload)
    print("> decode json")
    cmd = decode_result(payload)
    print(cmd)
    print("> encode bin")
    payload = cmd.encode_bin()
    print_bin_res(payload)
    print("> decode bin")
    cmd = decode_result(payload)
    print(cmd)

def main():
    print(">>> TEST ChangeWeightCommand")
    cmd = ChangeWeightCommand()
    cmd.add_weight(1, 2.222)
    cmd.add_weight(4, 5.555)
    cmd.add_weight(9, 12.3)
    test_cmd(cmd)
    print("")

    print(">>> TEST ChangeWeightResult")
    res = ChangeWeightResult(RES_OK)
    test_res(res)
    print("")

    print(">>> TEST EvalWeightCommand")
    cmd = EvalWeightCommand()
    cmd.add_task(1, 13, 0.077)
    cmd.add_task(2, 24, 0.088)
    cmd.add_task(3, 37, 12.3)
    test_cmd(cmd)
    print("")

    print(">>> TEST EvalWeightResult")
    res = EvalWeightResult(RES_OK)
    res.add_eval(1, 1.11)
    res.add_eval(2, 2.22)
    res.add_eval(3, 3.33)
    test_res(res)
    print("")

if __name__ == "__main__":
    main()

