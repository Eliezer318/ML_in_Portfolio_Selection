import torch
from torch import nn


class MyLSTM(nn.Module):
    def __init__(self, input_dim=503, hidden_size=503, num_layers=2, dropout=0.8):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, input_dim)

    def load_weights(self, path_weights: str):
        self.load_state_dict(torch.load(path_weights))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape is [B, window_size, D=503]
        if self.training:
            pass
        else:
            pass
        hiddens, _ = self.lstm(data)  # [B, T, hidden_size]
        x = self.linear(hiddens)      # [B, T, output_dim]
        x = x[:, -1, :]               # [B, output_dim]
        # x = x/x.sum(1).reshape(-1, 1)
        # return x.softmax(-1)
        return x.tanh()


class Weights(nn.Module):
    def __init__(self, D=503):
        super(Weights, self).__init__()
        # self.w = nn.Parameter()
        self.w = nn.Parameter(torch.Tensor(D, 1))
        nn.init.xavier_uniform_(self.w)

    def init_vector(self):
        weights = torch.tensor([0.001047831219645778, 0.00024168732616914465, 0.0003153293994568129, 0.070140766111528, 0.006260769872576045, 0.0007786664402206172, 0.00031693737368359036, 0.004825973882336999, 0.005068368022208586, 0.005149316488944081, 0.0022163162722636832, 0.001230938723339274, 0.002717308657763196, 0.0012461869291323366, 0.0006261519954301953, 0.001343765714752126, 0.00043556240855657633, 0.0010222689025221573, 0.0010925535209022996, 0.00022907077087098478, 0.0010095719190000684, 0.000385624437064433, 0.0008204862937203028, 0.0005513550303322993, 0.00014833127893684755, 0.0009010856906721718, 0.00024019088386999815, 0.002360535010258333, 0.00047905919309149407, 0.00406355250621522, 0.0007487162425202945, 0.0033475405249814533, 0.0007910978617576335, 0.0032145205574543377, 0.03633175054270016, 0.0010222895356721377, 0.0006107218894293442, 0.001599431076160682, 0.00024815189223039043, 0.0002986932807069763, 0.0014874940818141538, 0.0011946764241847132, 0.0007248352704888669, 0.0006782443346767718, 0.0004139794407726677, 0.0015770985142501803, 0.0007587715669299527, 0.005667940260385649, 0.000406265568275848, 0.0007232421038834064, 0.0030937595552852967, 0.0011331117240047936, 0.002583697617213255, 0.007305931922242751, 0.00048609273622763786, 0.0007705315385464588, 0.00023394350480910697, 0.00046510261222196286, 0.0018848552910329256, 0.00035545528160834685, 0.0009257203372665692, 0.000786778809673777, 0.00038605093248672925, 0.0009044716829712129, 0.002134226309867575, 0.0006380334049393728, 0.002815827534863192, 0.0039679982329532204, 0.0005207821426430123, 0.01679799720585978, 0.0004800107816502969, 0.0015295004035573198, 0.00023450663069221945, 0.0003423284151318754, 0.0025991634462797443, 0.00042507961352599094, 0.0004698763634990467, 0.0009343512557601596, 0.002610993118935239, 0.0021214095308926145, 0.00033576966541338024, 0.0006953594299331211, 0.0019648757792817342, 0.0003359159195448964, 0.00025496958284851665, 0.0013091005856500368, 0.0006341928674258355, 0.0003174578731212829, 0.0006684078140441457, 0.0005112161113148573, 0.000493635487027853, 0.0005401614951416264, 0.0003687465717066158, 0.00184431741312295, 0.002224351087762563, 0.0004235958488090464, 0.0017319310061862988, 0.0004590182966374102, 0.0002838908433407315, 0.004270383307719397, 0.0018729549934660545, 0.0011903711788604206, 0.0008134096365394236, 0.0005128770285620449, 0.0013657224659060173, 0.0005141521777613382, 0.001095135538919513, 0.0004042832717535285, 0.003346114887235536, 0.0062274308083070315, 0.0003825974205140578, 0.0007766648193675324, 0.00038396585516133514, 0.00028062911187776705, 0.004686285198614192, 0.005123581920946493, 0.0018433237370442863, 0.0011169283540509226, 0.0004774777288696045, 0.0006097321141452043, 0.0008970683855752193, 0.0011232398395902357, 0.00032813868480018873, 0.0034099296272882065, 0.007800950959496661, 0.00026416824658715554, 0.0017869923633278325, 0.0005542362290242301, 0.0007776848258092308, 0.0028187077583564915, 0.0007425980002886346, 0.001465806485482396, 0.00039732107209617307, 0.000669069819935809, 0.005364826352958469, 0.005604654591253375, 0.00024520630476569704, 0.0009646302493778336, 0.0009457420890317407, 0.0005052792047023047, 0.0010082728517711956, 0.0003749796307455056, 0.0006294734706408646, 0.0004066480514441435, 0.0006576075918591959, 0.0021724212474036443, 0.00020510779233363346, 0.001114591880783726, 0.00015111352063086186, 0.000854604849652883, 0.0009230859664807485, 0.0006683018766770816, 0.0012526152332023852, 0.0008969051681201496, 0.0006527122487107095, 0.0006862630337351195, 0.0025221050982092326, 0.0029579074056290153, 0.0003089490854246542, 0.0013380029067583145, 0.0009980637539353146, 0.0017476358102436068, 0.0006458244207805227, 0.0016009869799131927, 0.0007737390694960417, 0.0008158011828661452, 0.0004953960285141257, 0.0015334380503239991, 0.0006142057608751834, 0.0003631532173064876, 0.00040806096028161515, 0.001546065948722024, 0.0011510552761612313, 0.0004425906137633719, 0.0004488078129929483, 0.0007500047365429583, 0.001627174040075527, 0.0005653774114462957, 0.000803653852088189, 0.0002242700219896976, 0.0011186540450177007, 0.0004323502582450576, 0.001529877754101242, 0.0005898447858123685, 0.0002592866075457457, 0.0015282664180055001, 0.0017720087956474553, 0.0006416219305961829, 0.0004442525062091905, 0.00035636432071109534, 0.0004782244743896704, 0.0004791070804768966, 0.0007602285650107907, 0.00022901397838229464, 0.0010329405524665569, 0.0006091175850289936, 0.001674894923143396, 0.0021767973229442875, 0.0020474022200545447, 0.0011570340649886695, 0.00025645293695551134, 0.0007808812702901674, 0.001414872579556865, 0.00043131305751170873, 0.03937392681242422, 0.039405731016700654, 0.0005670126142453734, 0.0009324525953520024, 0.00047892805453875724, 0.0030328979219920267, 0.0007465524820632583, 0.0006780211681690223, 0.00028586102686953014, 0.0005179384634352502, 0.001546851034946158, 0.008376467613761117, 0.0009070810065500921, 0.0005652732191715185, 0.0002370534645693541, 0.0009539165119135732, 0.0004478556085192207, 0.0034395736126184204, 0.0004939859426200593, 0.0009130646200444412, 0.0006973906660288397, 0.00025942675385425766, 0.0003553703110118475, 0.0011848976455762863, 0.0015727900866987763, 0.00040018659062064356, 0.003123964433807021, 0.0015762153949005168, 0.000781906255377508, 0.00041165641498132913, 0.0007798251814991249, 0.0008241555555585043, 0.00041039371239614105, 0.0037250924730841068, 0.0033773994646204195, 0.0004004707583690692, 0.0002943861106485439, 0.0011162917033236631, 0.0005312882141251305, 0.0004010416345149853, 0.0020547819073784214, 0.0006243804701277638, 0.001680737286814993, 0.00021298084034382896, 0.00043208392636632896, 0.0004990627753663843, 0.0010172914887099423, 0.000371543621021947, 0.010988736908420525, 0.00024465547151795856, 0.008939934508436954, 0.0006470409554094733, 0.001420626251479292, 0.00045906469556174456, 0.0008088673666215738, 0.0011750234001545168, 0.00036239405083543833, 0.0013576383772131544, 0.00116005276668756, 0.0010738616321122403, 0.0003938700494463227, 0.0070686954539513685, 0.0008898408292377663, 0.0003559712130103472, 0.00034763503547139134, 0.0006094276981936261, 0.0005673446437160797, 0.0011431799826278723, 0.0038739833363932493, 0.0003869499116457061, 0.007542035928292418, 0.002919544323318478, 0.0002170187016013963, 0.00040354058100671743, 0.0034673349513285215, 0.0017272227471962177, 0.000286458438683456, 0.0005796452859840653, 0.0007223759221941746, 0.000295608701774255, 0.000745256597061497, 0.0005560025190498009, 0.008654890723683256, 0.0005521191754489906, 0.0013424806082615484, 0.0003181455421348126, 0.004915275387282516, 0.0010064770491554787, 0.0013275936365718761, 0.0014709757567936835, 0.0022515597454801518, 0.0031714590707925166, 0.0013594871485124033, 0.011764122883099215, 0.0003509204796589603, 0.0001944628833622186, 0.0006191814809213552, 0.00025708306924981673, 0.000574553158017108, 0.002164949994060949, 0.002093887783088257, 0.001205054488014962, 0.002058448038315755, 0.0004787450251536067, 0.0004735764210834878, 0.0012595057814234899, 0.000618246470739278, 0.005802817827325936, 0.0014706171916549673, 0.0004177410385233442, 0.003946828005158287, 0.0010087270890282295, 0.05423479767002382, 0.0010717012078610467, 0.0008414166739782729, 0.0004612697736449855, 0.0007663892027197737, 0.0017411923136054133, 0.00014584327776659143, 0.0007883919421880496, 0.000355131797957222, 0.004443473057813647, 0.0008881309467540576, 0.0027324349124011496, 0.0003226127217580815, 0.0045604332489818, 0.0003403732445293686, 0.00024961907857060935, 0.0019110495368693825, 0.002478576337424469, 0.0002564707728252084, 0.0015222044806632148, 0.00041571739871169806, 0.0005363528825516908, 0.0009359389817837846, 0.011724416900947658, 0.00036528438559875356, 0.0002192302211282257, 0.0002676506039088047, 0.00026778246102895383, 0.001202768622424086, 0.0011297023269386103, 0.0008496441168698813, 0.0001958707878909263, 0.0007201019129656007, 0.00037588967071000687, 0.000795319599283373, 0.005297586510009033, 0.001169071916888744, 0.000846854381544282, 0.0015146053223211375, 0.00042566686274365776, 0.0005821863969850238, 0.0012552038209787963, 0.0008210716695300734, 0.0003801255229527728, 0.0008675311588319693, 0.0001419459092696812, 0.0062395076681523425, 0.006833006163951062, 0.0004952682261672314, 0.008969133803191469, 0.0018794216895664153, 0.0009675446561494594, 0.00025359319263346076, 0.00034097391555977145, 0.0004687181354829763, 0.0025351817932824554, 0.0039043803803715084, 0.001783299132133808, 0.0002077305505568096, 0.00022061102540024262, 0.00038837898824995354, 0.0007861184975481568, 0.000570289692255214, 0.0009828233497312991, 0.0015566244756252368, 0.0010692403197790424, 0.0003594196180217672, 0.00011780543174994156, 0.0005178728684957599, 0.0014463592796895674, 0.0028942712809045344, 0.0042604732365795015, 0.0002768807306348517, 0.0002672578041655189, 0.0002787167730256551, 0.0002841770898020284, 0.0017069853225548817, 0.0005376657565401276, 0.00022236735813452926, 0.0006002829526740284, 0.00016852362630462542, 0.00086603756513935, 0.0007417628196506173, 0.0004589535229178197, 0.00115238041712192, 0.000806814316872269, 0.001166371027291046, 0.003504626546974896, 0.000931038659989656, 0.0003180235396534601, 0.0025463197934774097, 0.003601472599139159, 0.0004423020576211071, 0.00021331856702758962, 0.001617197552776242, 0.0006773607533909147, 0.00037205665249117644, 0.001316144804644877, 0.000309595462475101, 0.0014311891924394659, 0.002123782856402811, 0.0010641540943301143, 0.0032377444509135004, 0.0013425772042522533, 0.0005334419146383951, 0.0006836146508847599, 0.00042706719665141493, 0.0011955515366403543, 0.0003796013280255315, 0.0004466571407280639, 0.0004270457679446565, 0.0020938602722216162, 0.001100358086871795, 0.00329140029019666, 0.00030504274766665173, 0.0008909626156207865, 0.0004738888439290883, 0.0003553978732047319, 0.0010983901360345986, 0.0003965756097322004, 0.00171027584804058, 0.00028853232688771637, 0.0020110112973331902, 0.001996992868339961, 0.005842329180361259, 0.0046155605090202854, 0.00023708818677323923, 0.0004434060337975521, 0.0007453191124263634, 0.0010355110734051741, 0.0005708618772203452, 0.023784351404125725, 0.0007370703178173623, 0.0009695715295144606, 0.0005450450332540543, 0.0008411612232632156, 0.004086555286216383, 0.0003604910790228076, 0.00042289488629839, 0.0003198532175900407, 0.0006750878733397029, 0.00019798109206511499, 0.0005314711408577936, 0.01278117416498849, 0.003786004816706185, 0.00452929998136344, 0.0005669289524680892, 0.001824940524284035, 0.011368462420292767, 0.0004540047491496076, 0.0008467151847712791, 0.0011728772445990807, 0.0005921682222136569, 0.00015244946576585644, 0.0008021943903476518, 0.0005551334630909201, 0.001891485409893331, 0.00049105608664907, 0.000321822117962746, 0.004651069236264364, 0.00043219776797492795, 0.000496979853743227, 0.0008282669929864623, 0.0008024715007375642, 0.00038237915566258484, 0.0008490313328459575, 0.0009290250287955378, 0.0043860714292628805, 0.0002359326533849648, 0.001812700241678679, 0.0010761720316476207, 0.00955379227737963, 0.00045267432158581426, 0.000271558892064069, 0.0005971664744808272, 0.0006084263745046464, 0.0006809138126133055, 0.0001903899791117428, 0.0010554168278595099, 0.00985779310229851, 0.0001963287719637525, 0.00046217747826539693, 0.0008411120527217198, 0.0006021958304516995, 0.00045060017675033524, 0.00022170006563976142, 0.001996490897676256])
        self.w = nn.Parameter(weights/weights.sum())

    def forward(self, cov_matrix, returns):
        weights = self.w.tanh()
        future_returns = (weights.T @ returns)
        std = (weights.T @ cov_matrix @ weights) ** 0.5
        return weights, future_returns/std

