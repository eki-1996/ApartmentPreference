import time
from otree.api import *
from collections import OrderedDict
from experiments.bace.generate_set import BACE
import experiments.bace.user_config_5var_BACE1 as user_config_1
import experiments.bace.user_config_5var_BACE2 as user_config_2
import random
import json
import numpy as np

doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'experiments'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1

    bace_round = 30
    record_times = 3

    rest_time = 10 # in seconds

    show_digitals = 1

    def constract_selection_set(start, end, step, unit):
        ret = []
        if unit == "階":
            for i, index in enumerate(np.arange(start, end, step)):
                ret.append(f"{index}{unit}")
        else:
            ret.append(f"{start}{unit}未満")
            for i, index in enumerate(np.arange(start, end, step)):
                ret.append(f"{index}{unit}以上{index+step}{unit}未満")
        ret.append(f"{end}{unit}以上")
        return ret

    prior_setting = OrderedDict(
        price = [3, 10, .5, "万円"],
        room_size = [6, 20, 1, "畳"],
        distance_to_university = [5, 30, 5, "分"],
        city_center = ["繁華街", "住宅街"],
        floor = [1, 5, 1, "階"]
    )
    prior = OrderedDict()
    for k, v in prior_setting.items():
        # print(f"{k}: {constract_selection_set(*v)}")
        if k == "city_center":
            prior[k] = v
        else:
            prior[k] = constract_selection_set(*v)

    reference = dict(
        price = "家賃",
        room_size = "部屋の広さ",
        distance_to_university = "学校までの距離",
        city_center = "立地",
        floor = "階層",
    )

class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # 性別
    gender = models.IntegerField(
        label="あなたの性別を選んでください", 
        choices=[
            [1, '男性'],
            [2, '女性'],
            [3, '回答しない'],
        ],
    )
    # 年齢
    age = models.IntegerField(
        label="あなたの年齢を選んでください", 
        choices=range(15, 80),
    )
    # 学部
    belong = models.IntegerField(
        label="あなたの所属を選んでください", 
        choices=[
            [1, '文学部'],
            [2, '人間科学部'],
            [3, '外国語学部'],
            [4, '法学部'],
            [5, '経済学部'],
            [6, '理学部'],
            [7, '医学部医学科'],
            [8, '医学部保健学科　　　　　　　　　'],
            [9, '歯学部'],
            [10,'薬学部'],
            [11, '工学部'],
            [12, '基礎工学部'],
            [101, '人文学研究科（文学研究科）'],
            [102, '人間科学研究科'],
            [103, '法学研究科'],
            [104, '経済学研究科'],
            [105, '理学研究科'],
            [106, '医学系研究科'],
            [107, '歯学研究科'],
            [108, '薬学研究科'],
            [109, '工学研究科'],
            [110, '基礎工学研究科'],
            [111, '言語文化研究科'],
            [112, '国際公共政策研究科'],
            [113, '情報科学研究科'],
            [114, '生命機能研究科'],
            [115, '高等司法研究科'],
            [116, '連合小児発達学研究科'],
            [117, 'その他（附置研究所・センター等）'],
        ],
    )

    # 学年
    grade = models.IntegerField(
        label="あなたの学年を選んでください", 
        choices=[[1, 'B1'],
                 [2, 'B2'],
                 [3, 'B3'],
                 [4, 'B4'],
                 [11, 'M1'],
                 [12, 'M2'],
                 [21, 'D1'],
                 [22, 'D2'],
                 [23, 'D3'],
                 [30, 'その他']]
    )

    price = models.StringField(label="家賃", choices=C.prior["price"])
    room_size = models.StringField(label="部屋の広さ", choices=C.prior["room_size"])
    distance_to_university = models.StringField(label="学校までの距離", choices=C.prior["distance_to_university"])
    city_center = models.StringField(label="立地", choices=C.prior["city_center"])
    floor = models.StringField(label="階層", choices=C.prior["floor"])

    selection_set_bace_1 = models.StringField(initial="")
    posterior_bace_1 = models.LongStringField(initial="")
    answers_bace_1 = models.StringField(initial="")
    answer_times_bace_1 = models.StringField(initial="")

    selection_set_bace_2 = models.StringField(initial="")
    posterior_bace_2 = models.LongStringField(initial="")
    answers_bace_2 = models.StringField(initial="")
    answer_times_bace_2 = models.StringField(initial="")

    selection_set_eval = models.StringField(initial="")
    answers_eval = models.StringField(initial="")
    answer_times_eval = models.StringField(initial="")

    current_stage = models.StringField(initial="bace_1")
    start_time = models.FloatField()

# functions
def selection_set(player: Player, reverse=False, random=False, eval=False):
    already_generated = True
    if getattr(player, "selection_set_" + player.current_stage) == "":
        already_generated = False
    else:
        if not "&" in getattr(player, "selection_set_" + player.current_stage) and getattr(player, "answers_" + player.current_stage) == "":
            already_generated = True
        elif len(getattr(player, "selection_set_" + player.current_stage).split("&")) == len(getattr(player, "answers_" + player.current_stage).split("&")):
            already_generated = False
        elif len(getattr(player, "selection_set_" + player.current_stage).split("&")) > len(getattr(player, "answers_" + player.current_stage).split("&")):
            already_generated = True
        else:
            raise Exception(f"Player {player.id_in_group} in group {player.group.id_in_subsession} has more answers than provided selection set!")
    
    if already_generated:
        return json.loads(getattr(player, "selection_set_" + player.current_stage).split("&")[-1])
    player.start_time = time.time()
    assert player.group.id_in_subsession <= 2
    if player.group.id_in_subsession == 1:
        bace = BACE(user_config_1)
        if reverse: bace = BACE(user_config_2)
    if player.group.id_in_subsession == 2:
        bace = BACE(user_config_2)
        if reverse: bace = BACE(user_config_1)
    if eval:
        for k, v in user_config_2.design_params.items():
            if not k in user_config_1.design_params.keys():
                user_config_1.design_params[k] = v
        bace = BACE(user_config_1)
    
    if getattr(player, "selection_set_" + player.current_stage) != "":
        for i, selection_set in enumerate(getattr(player, "selection_set_" + player.current_stage).split("&")):
            bace.add_record(json.loads(selection_set), int(getattr(player, "answers_" + player.current_stage).split("&")[i]))
    
    ret, posterior = bace.interactive_bace_steps(random)
    # print(ret)
    if getattr(player, "selection_set_" + player.current_stage) == "":
        setattr(player, "selection_set_" + player.current_stage, json.dumps(ret, ensure_ascii=False))
    else:
        # player.selection_set += json.dumps(ret, ensure_ascii=False)
        setattr(player, "selection_set_" + player.current_stage, getattr(player, "selection_set_" + player.current_stage) + "&" + json.dumps(ret, ensure_ascii=False))
    
    if player.current_stage != "eval":
        if getattr(player, "posterior_" + player.current_stage) == "":
            if len(getattr(player, "selection_set_" + player.current_stage).split("&")) == int(C.bace_round / C.record_times):
                setattr(player, "posterior_" + player.current_stage, posterior.to_string())
        else:
            if len(getattr(player, "selection_set_" + player.current_stage).split("&")) == int(C.bace_round / C.record_times) * (len(getattr(player, "posterior_" + player.current_stage).split("&"))):
                setattr(player, "posterior_" + player.current_stage, getattr(player, "posterior_" + player.current_stage) + "&" + posterior.to_string())
    return ret

def convert_to_items(selection_set: dict):
    items = [dict(), dict()]
    for k, v in selection_set.items():
        if k[:-2] == "city_center":
            unit = ""
        else:
            unit = C.prior_setting[k[:-2]][-1]

        if k[:-2] == "floor":
            digit = 0
        else:
            digit = C.show_digitals
        if k[-1] == "a":
            if k[:-2] == "floor":
                items[0][C.reference[k[:-2]]] = str(round_floats(v, digit)).split(".")[0] + unit
            else:
                items[0][C.reference[k[:-2]]] = str(round_floats(v, digit)) + unit
        elif k[-1] == "b":
            if k[:-2] == "floor":
                items[1][C.reference[k[:-2]]] = str(round_floats(v, digit)).split(".")[0] + unit
            else:
                items[1][C.reference[k[:-2]]] = str(round_floats(v, digit)) + unit
    return items

def live_method(player: Player, data):
    if data:
        try:
            answer = data["answer"]
        except Exception:
            print("invalid message received:", data)
            
    answer_time = time.time()
    if getattr(player, "answers_" + player.current_stage) == "":
        setattr(player, "answers_" + player.current_stage, answer)
        setattr(player, "answer_times_" + player.current_stage, str(int(answer_time - player.start_time)))
    else:
        setattr(player, "answers_" + player.current_stage, getattr(player, "answers_" + player.current_stage) + "&" + answer)
        setattr(player, "answer_times_" + player.current_stage, getattr(player, "answer_times_" + player.current_stage) + "&" + str(int(answer_time - player.start_time)))
    
    player.start_time = answer_time
    return {
        player.id_in_group: dict(
            next = True,
            submit = True if len(getattr(player, "answers_" + player.current_stage).split("&")) == C.bace_round else False,
        )
    }

def round_floats(data, digits=C.show_digitals):
    if isinstance(data, float):
        return round(data, digits)
    elif isinstance(data, dict):
        return {k: round_floats(v, digits) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_floats(item, digits) for item in data]
    else:
        return data    

# PAGES
class Start(Page):
    pass

class BasicInfo(Page):
    form_model = 'player'
    form_fields = ["gender", "age", "belong", "grade"]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1
    
class ResidenceInfo(Page):
    form_model = 'player'
    form_fields = ["price", "room_size", "distance_to_university", "city_center", "floor"]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

class Select1(Page):
    live_method = live_method

    @staticmethod
    def vars_for_template(player: Player):
        player.current_stage = "bace_1"
        return dict(
            items = convert_to_items(selection_set(player)),
        )

class Rest1(Page):
    timeout_seconds = C.rest_time

class Select2(Page):
    live_method = live_method

    @staticmethod
    def vars_for_template(player: Player):
        player.current_stage = "bace_2"
        return dict(
            items = convert_to_items(selection_set(player, reverse=True)),
        )
    
class Rest2(Page):
    timeout_seconds = C.rest_time

class Eval(Page):
    live_method = live_method

    @staticmethod
    def vars_for_template(player: Player):
        player.current_stage = "eval"
        return dict(
            items = convert_to_items(selection_set(player, random=True, eval=True)),
        )
    
class End(Page):
    pass


page_sequence = [Start, BasicInfo, ResidenceInfo, Select1, Rest1, Select2, Rest2, Eval, End]
