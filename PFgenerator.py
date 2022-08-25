###====================================================================================================
### License
###====================================================================================================
# PFgenerator is a program for generating artificial protein families by simulating protein evolution.

# Copyright (C) 2018,   Lucas Carrijo de Oliveira (lucas.carrijodeoliveira@gmail.com)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
###====================================================================================================
### Libraries
###====================================================================================================
try:
	import argparse
	import numpy as np
	from scipy.stats import gamma
	import string
	import networkx as nx
	from Bio.Seq import Seq
	from Bio.SeqRecord import SeqRecord
	from Bio.Alphabet import IUPAC
	from Bio import AlignIO
	from Bio import Phylo
	# from weblogolib import *
	from matplotlib import pyplot as plt
	from Bio.SeqUtils import seq3
	# import pandas as pd
	from pandas import DataFrame
except:
	raise ImportError('Need the following packagens installed: argparse, numpy, scipy, matplotlib, networkx, Bio, weblogo, string and pandas.')	
###====================================================================================================
### Parameters
###====================================================================================================
parser=argparse.ArgumentParser(description="PFgenerator is a program for generating artificial protein families by simulating protein evolution.")
parser.add_argument("size", help="Minimum amount of 'current' sequences, i.e.,'leaves' of the phylogeny tree (sequences corresponding to internal nodes of the phylogeny tree will also be generated during the process.)", type=int)
parser.add_argument("length", help="Length of root sequence.", type=int)
parser.add_argument("-o", "--out", help="Label to be used for naming output files (default: stdout).", type=str, default='PFgenerator_test', required=False)
# parser.add_argument("-g", "--p_gap", help="Prior probability of occurring indels (default: 1/100).", type=float, default=.01, required=False)
parser.add_argument("-g", "--p_gap", help="Prior probability of occurring indels (default: 1/SIZE).", type=float, default=None, required=False)
parser.add_argument("-n", "--n_ort", help="Number of ortholog groups (default: 1).", type=int, default=1, required=False)
parser.add_argument("-a", "--shape", help="Shape parameter for gamma-distributed probabilities of mutation at each site. (default: 2.0).", type=float, default=2., required=False)
parser.add_argument("-b", "--scale", help="Scale parameter for gamma-distributed probabilities of mutation at each site. (default: 1.0).", type=float, default=1., required=False)
parser.add_argument("-r", "--seed", help="An integer to be used as seed for random operations (default: None).", type=int, default=None, required=False)
parser.add_argument("-f", "--msa_format", help="The file format for output sequences ('fasta', 'clustal', 'nexus', 'phylip' or 'stockholm'; default: 'fasta').", type=str, default='fasta', required=False)
parser.add_argument("-t", "--tree_format", help="The file format for output phylogeny tree ('newick', 'nexus', 'nexml', 'phyloxml' or 'cdao'; default: 'newick').", type=str, default='newick', required=False)
args = parser.parse_args()	# returns data from the options specified (echo)
if args.p_gap == None:
	args.p_gap = 1./float(args.size)
if args.msa_format not in ['fasta', 'clustal', 'nexus', 'phylip', 'stockholm']:
	raise ValueError("MSA_FORMAT must be 'fasta', 'clustal', 'nexus', 'phylip' or 'stockholm'!")
if args.tree_format not in ['newick', 'nexus', 'nexml', 'phyloxml', 'cdao']:
	raise ValueError("TREE_FORMAT must be 'newick', 'nexus', 'nexml', 'phyloxml' or 'cdao'!")
np.random.seed(args.seed)
###====================================================================================================
### Canonical amino acids.
amino_acids=['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
###====================================================================================================
### Occurrence of each amino acid in nature (Whelan, S. and N. Goldman.  2001.  A general empirical model of protein evolution derived from multiple protein families using a maximum likelihood approach.  Molecular Biology and Evolution 18:691-699.).
aa_freq={
	'A': 0.08662790866279087, 
	'R': 0.04397200439720044, 
	'N': 0.0390894039089404, 
	'D': 0.057045105704510574, 
	'C': 0.019307801930780195, 
	'Q': 0.03672810367281037, 
	'E': 0.05805890580589058, 
	'G': 0.08325180832518084, 
	'H': 0.024431302443130246, 
	'I': 0.04846600484660049, 
	'L': 0.08620900862090086, 
	'K': 0.062028606202860624, 
	'M': 0.019502701950270197, 
	'F': 0.03843190384319038, 
	'P': 0.045763104576310464, 
	'S': 0.06951790695179069, 
	'T': 0.06101270610127062, 
	'W': 0.014385901438590145, 
	'Y': 0.035274203527420354, 
	'V': 0.07089560708956072
}
###====================================================================================================
### Different sets of amino acids according to their physico-chemical properties.
sthereochemical_sets={
	'Amide':['N','Q'],
	'Aliphatic':['G','A','V','L','I'],
	'Basic':['H','K','R'],
	'With hydroxyl':['S','T','Y'],
	'With sulfur':['C','M'],
	'Non-polar':['F','G','V','L','A','I','P','M','W'],
	'Polar':['Y','S','N','T','Q','C'],
	'Very hydrophobic':['L','I','F','W','V','M'],
	'Hydrophilic':['R','K','N','Q','P','D'],
	'Positively charged':['K','R'],
	'Negatively charged':['D','E'],
	'Tiny':['G','A','S'],
	'Small':['C','D','P','N','T'],
	'Median':['E','V','Q','H'],
	'Big':['M','I','L','K','R'],
	'Aromatic':['F','Y','W'],
	'Similar (Asn or Asp)':['N','D'],
	'Similar (Gln or Glu)':['Q','E']
}
###====================================================================================================
# aux={a:np.mean(list(map(lambda x: aa_freq[x], sthereochemical_sets[a]))) for a in sthereochemical_sets}
aux={a:sum(list(map(lambda x: aa_freq[x], sthereochemical_sets[a]))) for a in sthereochemical_sets}
sc_freq={a:float(aux[a])/float(sum(aux.values())) for a in aux}
###====================================================================================================
### WAG Matrix for conditional probabilities of amino acid substitution (Whelan, S. and N. Goldman.  2001.  A general empirical model of protein evolution derived from multiple protein families using a maximum likelihood approach.  Molecular Biology and Evolution 18:691-699.).
aa_subs_matrix = {
	'A': {'R': 0.017296676952630402, 'N': 0.014046322866055283, 'D': 0.03762854110703619, 'C': 0.006258826188005301, 'Q': 0.019734028416788126, 'E': 0.08025812526560464, 'G': 0.1476468794503392, 'H': 0.0032510617996782966, 'I': 0.009506566406970974, 'L': 0.04491658382027503, 'K': 0.05293814961165839, 'M': 0.005647187018816816, 'F': 0.005159909931689336, 'P': 0.046500937632681705, 'S': 0.2372461131491415, 'T': 0.11912834453498683, 'W': 0.0004189929652447363, 'Y': 0.004817589060618512, 'V': 0.14759916382177868} ,
	'R': {'A': 0.04292698082856624, 'N': 0.021229250621732972, 'D': 0.01059852079190627, 'C': 0.004090013642190552, 'Q': 0.08138950143200809, 'E': 0.03136788892080954, 'G': 0.07932293412622492, 'H': 0.025529802185711173, 'I': 0.009633231115902217, 'L': 0.07123096972090054, 'K': 0.3911639891830717, 'M': 0.005397246252499383, 'F': 0.0033045491627062486, 'P': 0.02859414339357293, 'S': 0.1128632606460379, 'T': 0.04185933263483028, 'W': 0.004783323642666797, 'Y': 0.009514853234044036, 'V': 0.025200208464618304} ,
	'N': {'A': 0.02535025030497468, 'R': 0.015437842255801979, 'D': 0.2189054722863498, 'C': 0.0015621442633370643, 'Q': 0.027354993296306234, 'E': 0.04287438819972256, 'G': 0.0984256306286194, 'H': 0.030117891833286257, 'I': 0.01673843891464592, 'L': 0.012920979761018834, 'K': 0.14253272781700618, 'M': 0.0012157824160508707, 'F': 0.0021389773456134374, 'P': 0.006182047079204025, 'S': 0.23245435076339258, 'T': 0.09511901029995332, 'W': 0.00023029266157963008, 'Y': 0.01723138131778069, 'V': 0.013207398555356674} ,
	'D': {'A': 0.06659765454077526, 'R': 0.007558213224024489, 'N': 0.21467356424030978, 'C': 0.00037469637959860704, 'Q': 0.017541793059255222, 'E': 0.363990978854939, 'G': 0.10814554133521387, 'H': 0.010443047573839605, 'I': 0.002243573310191295, 'L': 0.011732967961938049, 'K': 0.03575169413671663, 'M': 0.000859061428620298, 'F': 0.0014376953442242857, 'P': 0.016649393820032474, 'S': 0.09248279741913484, 'T': 0.026958108593937408, 'W': 0.0005136829529454174, 'Y': 0.007554013994450562, 'V': 0.014491521829852787} ,
	'C': {'A': 0.07437463831337775, 'R': 0.019583443623239615, 'N': 0.010285691845926903, 'D': 0.002515764120042383, 'Q': 0.006421999817345728, 'E': 0.004215823502706541, 'G': 0.08910811478937682, 'H': 0.006615131901227128, 'I': 0.020285518924755212, 'L': 0.11768602133492684, 'K': 0.013027634009185059, 'M': 0.006847801829182283, 'F': 0.025266597306689054, 'P': 0.010688057825216531, 'S': 0.2731522532370249, 'T': 0.08104909557858522, 'W': 0.006341274721948706, 'Y': 0.028621396402788334, 'V': 0.20391374091645503} ,
	'Q': {'A': 0.04347821835864639, 'R': 0.07225313708655884, 'N': 0.03339435024003139, 'D': 0.02183677100269442, 'C': 0.001190677928689799, 'E': 0.26098824299199075, 'G': 0.03416620287268939, 'H': 0.0375055525953192, 'I': 0.005058292166424973, 'L': 0.09149185162118714, 'K': 0.21246947147749412, 'M': 0.008793652478951898, 'F': 0.002455533856360346, 'P': 0.02897378717760419, 'S': 0.07117977108337238, 'T': 0.04706598706755119, 'W': 0.0006892431336197038, 'Y': 0.004372051482769478, 'V': 0.022637205378044516} ,
	'E': {'A': 0.111358042076406, 'R': 0.017536793849207928, 'N': 0.03296179897314367, 'D': 0.2853524739395861, 'C': 0.0004922469348325616, 'Q': 0.16436052851873528, 'G': 0.05706985984180094, 'H': 0.005359279757682246, 'I': 0.00511989552646706, 'L': 0.01710935610559134, 'K': 0.13889345141807877, 'M': 0.001950311180493939, 'F': 0.001887601581177043, 'P': 0.02101773680158353, 'S': 0.04947371691580635, 'T': 0.044272122949658196, 'W': 0.0004942206341712699, 'Y': 0.0036751719579165482, 'V': 0.04161539103766116} ,
	'G': {'A': 0.2792690477297492, 'R': 0.06045463893302231, 'N': 0.10315425163062898, 'D': 0.11557545101696463, 'C': 0.014183513017255069, 'Q': 0.02933183659180943, 'E': 0.07779877559354606, 'H': 0.0046248051011830976, 'I': 0.002763831247447918, 'L': 0.013594790897813964, 'K': 0.041889995460642045, 'M': 0.002087745498274858, 'F': 0.0023454359084500317, 'P': 0.015445338453506237, 'S': 0.17775320390105098, 'T': 0.02641756864461632, 'W': 0.00200796997677795, 'Y': 0.0039118307107869855, 'V': 0.027389969686473607} ,
	'H': {'A': 0.021314512449627968, 'R': 0.06744192492846272, 'N': 0.1094095273926164, 'D': 0.03868440137268194, 'C': 0.0036496961577166375, 'Q': 0.11160656876816194, 'E': 0.025323517516555014, 'G': 0.016030426663727292, 'I': 0.010588340987023282, 'L': 0.10567116665570425, 'K': 0.09903885210763883, 'M': 0.004795379160775072, 'F': 0.029993805744459343, 'P': 0.04293362237571137, 'S': 0.10194982896767442, 'T': 0.051894627393405744, 'W': 0.0017155231403319302, 'Y': 0.13985649667826464, 'V': 0.018101781539461244} ,
	'I': {'A': 0.01424848326338812, 'R': 0.005817672372859849, 'N': 0.013900816071452299, 'D': 0.0018999565168644438, 'C': 0.0025585805642743993, 'Q': 0.003441064956168042, 'E': 0.0055306155948275345, 'G': 0.002190069723142795, 'H': 0.0024205981182118916, 'L': 0.2977162088754153, 'K': 0.016490862205360515, 'M': 0.022042641670891622, 'F': 0.021151615615115398, 'P': 0.003265617457644875, 'S': 0.02054034505313046, 'T': 0.07019795204940467, 'W': 0.0006987440579067104, 'Y': 0.007135445029213295, 'V': 0.4887527108047277} ,
	'L': {'A': 0.0671272542244438, 'R': 0.042893684357377934, 'N': 0.010699611434868778, 'D': 0.009907372594945361, 'C': 0.014800795907675425, 'Q': 0.062060974671045814, 'E': 0.018428638676953775, 'G': 0.010741530488205277, 'H': 0.024087875033038584, 'I': 0.2968586408629956, 'K': 0.02342381979029186, 'M': 0.04253650059427129, 'F': 0.07108186480515068, 'P': 0.020186242811958948, 'S': 0.03787112555446997, 'T': 0.029206740109865466, 'W': 0.0032174931375556744, 'Y': 0.011781696666644996, 'V': 0.2030881382782405} ,
	'K': {'A': 0.07381894801318682, 'R': 0.21978107391019694, 'N': 0.11012707047913389, 'D': 0.028167880353639017, 'C': 0.0015287366178710772, 'Q': 0.13447440838390734, 'E': 0.13958806572462218, 'G': 0.03088238810019216, 'H': 0.021064662975720804, 'I': 0.015342548587131953, 'L': 0.021855696336361224, 'M': 0.005668222676220259, 'F': 0.002208661986496115, 'P': 0.018504497425229933, 'S': 0.07140391708599818, 'T': 0.0782961863856377, 'W': 0.00048489446099862164, 'Y': 0.002823612518796695, 'V': 0.023978527978659108} ,
	'M': {'A': 0.0409536191646323, 'R': 0.01577119785216599, 'N': 0.004885357721154006, 'D': 0.003520000969118804, 'C': 0.004179066911863038, 'Q': 0.028944991422964153, 'E': 0.010193690693698, 'G': 0.008004590904348442, 'H': 0.005304358768356138, 'I': 0.10665440745232711, 'L': 0.20640951345232633, 'K': 0.029478672699017924, 'F': 0.045874663246134297, 'P': 0.00988918034496073, 'S': 0.060869159603931966, 'T': 0.14309678346890556, 'W': 0.002854290005895587, 'Y': 0.014119515143108809, 'V': 0.2589969401750908} ,
	'F': {'A': 0.026640947408341976, 'R': 0.00687467372948499, 'N': 0.006119192619862094, 'D': 0.004194042789111887, 'C': 0.010977977272168309, 'Q': 0.00575436599216514, 'E': 0.007024012911344399, 'G': 0.006402249317744073, 'H': 0.02362049207505725, 'I': 0.0728628474804039, 'L': 0.24556936774698848, 'K': 0.008177822991307693, 'M': 0.032660311062372153, 'P': 0.0126727083452266, 'S': 0.0915130636187306, 'T': 0.024149123074708564, 'W': 0.01167305311397303, 'Y': 0.2874728454507075, 'V': 0.11564090300030146} ,
	'P': {'A': 0.18589209372977772, 'R': 0.046058376920861725, 'N': 0.013693419468344126, 'D': 0.03760590819037205, 'C': 0.003595555759062934, 'Q': 0.052571271930289674, 'E': 0.06055535892629352, 'G': 0.03264360444985337, 'H': 0.026178608796388957, 'I': 0.008710028332720826, 'L': 0.05399611788655614, 'K': 0.05304902526685585, 'M': 0.005451291290037019, 'F': 0.009812080546938355, 'S': 0.2490056447671839, 'T': 0.09864779783893789, 'W': 0.0010105725417362099, 'Y': 0.009220505570090456, 'V': 0.0523027377876991} ,
	'S': {'A': 0.2228098715898064, 'R': 0.042709082827428115, 'N': 0.12096323596233664, 'D': 0.04907436926464854, 'C': 0.02158778747852215, 'Q': 0.030341421184995717, 'E': 0.03348708052793253, 'G': 0.08825802973216618, 'H': 0.014603994003940574, 'I': 0.012870579660037624, 'L': 0.023798597837015222, 'K': 0.04809038617086674, 'M': 0.007882654853945605, 'F': 0.01664603765307857, 'P': 0.05849857751537126, 'T': 0.18059497454590817, 'W': 0.0012662538814070471, 'Y': 0.0113358782651523, 'V': 0.015181187045440638} ,
	'T': {'A': 0.16358887745879075, 'R': 0.02316131939420562, 'N': 0.07237463449107338, 'D': 0.020916380342233924, 'C': 0.009366015300077276, 'Q': 0.02933525357863136, 'E': 0.04381637431025134, 'G': 0.019179309989306052, 'H': 0.010869533009733099, 'I': 0.06431587130743786, 'L': 0.026836729379590772, 'K': 0.07710458073571146, 'M': 0.027096201959382782, 'F': 0.0064229196078132746, 'P': 0.03388651067120696, 'S': 0.2640638824352941, 'W': 0.00039481179321367797, 'Y': 0.005696498244597628, 'V': 0.10157429599144868} ,
	'W': {'A': 0.016381430857798883, 'R': 0.0753541539416013, 'N': 0.004988909155422373, 'D': 0.01134746107278134, 'C': 0.020863621461288326, 'Q': 0.012230988558459563, 'E': 0.013926224750886663, 'G': 0.041505321765472565, 'H': 0.010230373847814216, 'I': 0.018227129497007097, 'L': 0.0841725658346142, 'K': 0.013595438826420834, 'M': 0.015388032036617615, 'F': 0.0883938436987115, 'P': 0.00988355962507904, 'S': 0.052714557953794705, 'T': 0.011240782712812458, 'Y': 0.31711444325678323, 'V': 0.182441161146634} ,
	'Y': {'A': 0.029771623155076015, 'R': 0.023692297895962427, 'N': 0.05900288524782113, 'D': 0.026376003545442644, 'C': 0.014884409301978808, 'Q': 0.012263149938710444, 'E': 0.016368837966789677, 'G': 0.012780689421560057, 'H': 0.13182726090560176, 'I': 0.029420420484801896, 'L': 0.048717885816546264, 'K': 0.012513499050914447, 'M': 0.012031850522246203, 'F': 0.34408207178479316, 'P': 0.014253722080322972, 'S': 0.07459206189497052, 'T': 0.025635515992037186, 'W': 0.05012376465292183, 'V': 0.061662050341502364} ,
	'V': {'A': 0.1634391265895826, 'R': 0.01124368315793988, 'N': 0.008103447339722042, 'D': 0.009066602534806739, 'C': 0.019001446011746553, 'Q': 0.011377293365469976, 'E': 0.033211886106132196, 'G': 0.016034859377442406, 'H': 0.0030573367529195244, 'I': 0.3610907509646727, 'L': 0.15047506133343697, 'K': 0.019041254575434624, 'M': 0.0395463724397988, 'F': 0.024801392511482626, 'P': 0.014487630109713009, 'S': 0.01789956779241879, 'T': 0.08190629683001363, 'W': 0.005167135632880849, 'Y': 0.011048856574385951} 
}
###====================================================================================================
aux={}
for a in sthereochemical_sets:
	aux[a]={}
	for b in sthereochemical_sets:
		if b!=a:
			temp=[]
			for x in sthereochemical_sets[a]:
				for y in sthereochemical_sets[b]:
					if y!=x:
						temp.append(aa_subs_matrix[x][y])
			aux[a][b]=np.mean(temp)
sc_subs_matrix={a:{b:float(aux[a][b])/float(sum(aux[a].values())) for b in aux[a]} for a in aux}
###====================================================================================================
def wave_shuffle(items):
	i=np.random.choice(range(len(items)))
	aux=[items.pop(i)]
	invert=False
	while len(items)>0:
		x=aux[-1]
		temp=np.array([abs(x-y) for y in items])
		temp=1.-temp
		p=temp/float(sum(temp))
		i=np.random.choice(range(len(items)), p=p)
		aux.append(items.pop(i))
	return aux
###====================================================================================================
### Assigns for each position a probability of occurring mutations following a gamma distribution.
def mutation_p_dist(length, duplication=False):
	if duplication:
		a,b=0,0
		while a<1:
			a=np.random.normal(args.shape, 1.)
		while b<1:
			b=np.random.normal(args.scale, 1.)
	else:
		a,b=args.shape,args.scale
	x=np.linspace(gamma.ppf(.01, a), gamma.ppf(.99, a), length-1)
	return [0.]+wave_shuffle(list(gamma.pdf(x,a, scale=b)))
###====================================================================================================
### Colects sequence objects in tree order from a Directed Acyclic Graph.
def build_MSA(graph, node): # A recursive function that takes as input the graph above and returns a dendrogram in Newick format.
	if graph.out_degree(node)==0:
		return None
	leaves=[]
	for child in graph.successors(node):
		if graph.out_degree(child)==0:
			leaves.append(SeqRecord(Seq(child.sequence, IUPAC.protein), id=str(child)))
		else:
			leaves+=build_MSA(graph, child)
	return leaves
###====================================================================================================
### Calculates the branch lengths of the phylogeny tree.
def get_branch_length(graph, node):
	if graph.in_degree(node)==0:
		return None
	parent=list(graph.predecessors(node))[0]
	mismatch = 0
	for x,y in zip(parent.sequence, node.sequence):
		if y!=x:
			mismatch+=1
	return float(mismatch)/float(args.length)
###====================================================================================================
### Builds a Clade object from a Directed Acyclic Graph.
def build_tree(graph, node, kind='gene'):
	if kind!='specie' and kind!='gene':
		return None
	clades=[]
	for child in graph.successors(node):
		if graph.out_degree(child)==0:
			if kind=='gene':
				clades.append(Phylo.BaseTree.Clade(name=str(child), branch_length=get_branch_length(graph, child)))
			else:
				clades.append(Phylo.BaseTree.Clade(name=str(child)))
		else:
			clades.append(build_tree(graph, child, kind))
	if kind=='gene':
		return Phylo.BaseTree.Clade(name=str(node), clades=clades, branch_length=get_branch_length(graph, node))
	return Phylo.BaseTree.Clade(name=str(node), clades=clades)
###====================================================================================================
def og_label(idx):
	a=list(string.ascii_uppercase)
	count=0
	if idx>=len(a):
		for i in range(idx+1):
			if i%len(a)==0:
				count+=1
		return '%s%d'%(a[idx%len(a)], count)
	return a[idx%len(a)]
###====================================================================================================
### Class for sequences.
class Sequence(object):
	def __init__(self, sequence, p_mutation, sthereochemistry, host, idx):
		super(Sequence, self).__init__()
		self.sequence=sequence # string containing the amino acid sequence.
		self.p_mutation=p_mutation # list containing the mutation probabilities for each position.
		self.sthereochemistry=sthereochemistry
		self.host=host
		self.idx=idx
		self.label=self.label_seq()
	def label_seq(self):
		return '%s_seq%d'%(self.host, self.idx+1)
	def mutant(self, new_host, new_idx, duplication=False): # a method for generating a new sequence by mutating a pre-existing sequence.
		current_sequence=list(self.sequence)[:]
		if duplication:
			ranked_positions=sorted(range(len(self.sequence)), key=lambda i: self.p_mutation[i])
			current_p_mutation=np.array(list(zip(*sorted(zip(ranked_positions, sorted(mutation_p_dist(len(self.sequence), True))), key=lambda x: x[0])))[1])
			current_sthereochemistry=[self.sthereochemistry[0]]
			for i,x in enumerate(current_p_mutation):
				if i>0:
					if np.random.choice((True, False), p=(1.-x, x)):
						if self.sthereochemistry[i]==None:
							current_sthereochemistry.append(np.random.choice(list(sc_freq.keys()), p=list(sc_freq.values())))
						else:
							p={k:current_p_mutation[i]*v for k,v in sc_subs_matrix[self.sthereochemistry[i]].items()}
							p[self.sthereochemistry[i]]=1.-current_p_mutation[i]
							current_sthereochemistry.append(np.random.choice(list(p.keys()), p=list(p.values())))
						if current_sequence[i] not in sthereochemical_sets[current_sthereochemistry[i]]:
							if current_sequence[i]=='-':
								p={k:v for k,v in aa_freq.items() if k in sthereochemical_sets[current_sthereochemistry[i]]}
							else:
								p={k:v for k,v in aa_subs_matrix[current_sequence[i]].items() if k in sthereochemical_sets[current_sthereochemistry[i]]}
							p={k:float(v)/float(sum(p.values())) for k,v in p.items()}
							current_sequence[i]=np.random.choice(list(p.keys()), p=list(p.values()))
					else:
						current_sthereochemistry.append(None)
		else: 
			current_p_mutation=list(self.p_mutation)[:]
			current_sthereochemistry=self.sthereochemistry[:]
		new_sequence, new_p_mutation, new_sthereochemistry, gaps, i = '', [], [], [], 0
		while i<len(current_sequence):
			if current_sequence[i]=='-':
				start,end=i,len(current_sequence)
				while i<end:
					if current_sequence[i]!='-':
						end=i
					i+=1
				i=start
				j, done = 0, False
				while i<end:
					if done:
						new_sequence+='-'
					else:
						p=args.p_gap*(2.**(-float(j)))
						if np.random.choice((True, False), p=(p, 1.-p)):
							if current_sthereochemistry[i]==None:
								new_sequence+=np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values()))
							else:
								p={k:v for k,v in aa_freq.items() if k in sthereochemical_sets[current_sthereochemistry[i]]}
								p={k:float(v)/float(sum(p.values())) for k,v in p.items()}
								new_sequence+=np.random.choice(list(p.keys()), p=list(p.values()))
						else:
							new_sequence+='-'
							done=True
					new_p_mutation.append(current_p_mutation[i])
					new_sthereochemistry.append(current_sthereochemistry[i])
					i+=1
					j+=1
			else:
				p=args.p_gap*current_p_mutation[i]
				if np.random.choice((True, False), p=(p, 1.-p)):
					new_sequence+='-'
					new_p_mutation.append(current_p_mutation[i])
					new_sthereochemistry.append(current_sthereochemistry[i])
				else:
					if current_sthereochemistry[i]==None:
						p={k:v*current_p_mutation[i] for k,v in aa_subs_matrix[current_sequence[i]].items()}
						p[current_sequence[i]]=1.-current_p_mutation[i]
					else:
						p={k:v for k,v in aa_subs_matrix[current_sequence[i]].items() if k in sthereochemical_sets[current_sthereochemistry[i]]}
						p={k:float(v)*current_p_mutation[i]/float(sum(p.values())) for k,v in p.items()}
						p[current_sequence[i]]=1.-current_p_mutation[i]
					new_sequence+=np.random.choice(list(p.keys()), p=list(p.values()))
					new_p_mutation.append(current_p_mutation[i])
					new_sthereochemistry.append(current_sthereochemistry[i])
					j, done = 0, False
					while not done:
						p=current_p_mutation[i]*args.p_gap*(2.**(-float(j)))
						if np.random.choice((True, False), p=(p, 1.-p)):
							new_sequence+=np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values()))
							new_p_mutation.append(1.)
							new_sthereochemistry.append(None)
							insertion=False
							if i<len(current_sequence)-1:
								if current_sequence[i+1]=='-':
									i+=1
								else:
									insertion=True
							else:
								insertion=True
							if insertion:
								gaps.append(i)
						else:
							done=True
						j+=1					
				i+=1	
		sequences.align(gaps)
		return Sequence(new_sequence, new_p_mutation, new_sthereochemistry, new_host, new_idx)
	def __repr__(self):
		return self.label
###====================================================================================================
### Class for species.
class Specie(object):
	def __init__(self, paralogs, label=''):
		super(Specie, self).__init__()
		self.paralogs=paralogs # list of paralogs.
		self.label=label
	def speciation(self): # method for simulating a speciation event.
		a,b=Specie([], 'sp%d'%len(sp_tree.nodes())), Specie([], 'sp%d'%(len(sp_tree.nodes())+1))
		for i,x in enumerate(self.paralogs):
			sequences.colection.append(x.mutant(a, i))
			a.paralogs.append(sequences.colection[-1])
			seq_tree.add_edge(self.paralogs[i], a.paralogs[i])
			sequences.colection.append(x.mutant(b, i))
			b.paralogs.append(sequences.colection[-1])
			seq_tree.add_edge(self.paralogs[i], b.paralogs[i])
		sp_tree.add_edge(self, a)
		sp_tree.add_edge(self, b)
	def gene_duplication(self):
		if args.n_ort>1:
			if len(orthologs)<args.n_ort:
				p=2**(-float(args.n_ort-len(orthologs))/float(len(current_species)))
				duplication=np.random.choice((True, False), p=(p, 1.-p))
				if duplication:
					if len(current_species)==1:
						i=0
					else:
						i=np.random.choice(range(len(self.paralogs)))
					sequences.colection.append(self.paralogs[i].mutant(self, len(self.paralogs), duplication))
					self.paralogs.append(sequences.colection[-1])
					orthologs.append(sequences.colection[-1])
					seq_tree.add_edge(self.paralogs[i], self.paralogs[-1])
	def __repr__(self):
		return self.label
###====================================================================================================
### Class for family.
class Family(object):
	def __init__(self, sequences, label=args.out):
		super(Family, self).__init__()
		self.colection=sequences # list of all sequences, including internal nodes.
		self.label=label
	def align(self, gaps):
		for i in sorted(gaps, reverse=True):
			for seq in self.colection:
				seq.sequence=seq.sequence[:i+1]+'-'+seq.sequence[i+1:]
				seq.p_mutation=seq.p_mutation[:i+1]+[1.]+seq.p_mutation[i+1:]
				seq.sthereochemistry=seq.sthereochemistry[:i+1]+[None]+seq.sthereochemistry[i+1:]
		args.length+=len(gaps)
	def __repr__(self):
		return self.label
###====================================================================================================
### MAIN
###====================================================================================================
if __name__ == '__main__':
	first_sp=Specie([], 'sp0')
	sp_tree=nx.DiGraph() # phylogeny tree of the "species" represented as a directed acyclic graph.
	seq_tree=nx.DiGraph() # phylogeny tree of the sequences represented as a directed acyclic graph.
	p_mutation=mutation_p_dist(args.length) # initial probabilities of mutation for each position.
	sequence, sthereochemistry = 'M', ['With sulfur']
	for i in range(1,args.length):
		if np.random.choice((True, False), p=(1.-p_mutation[i], p_mutation[i])):
			sthereochemistry.append(np.random.choice(list(sc_freq.keys()), p=list(sc_freq.values())))
			p={x:aa_freq[x] for x in sthereochemical_sets[sthereochemistry[i]]}
			p={x:float(p[x])/float(sum(p.values())) for x in p}
			sequence+=np.random.choice(list(p.keys()), p=list(p.values()))
		else:
			sthereochemistry.append(None)
			sequence+=np.random.choice(list(aa_freq.keys()), p=list(aa_freq.values()))
	first_seq=Sequence(sequence, p_mutation, sthereochemistry, first_sp, 0)
	sequences=Family([first_seq])
	first_sp.paralogs.append(sequences.colection[0])
	sp_tree.add_node(first_sp)
	current_species=list(sp_tree.nodes()) # at each iteration it stores the "leaves" of the species tree, from which the new species of the next iteration will be originated.
	orthologs=[first_seq]
	leaves=[x for x in seq_tree.nodes() if seq_tree.out_degree(x)==0] # stores the "leaves" in the sequence tree and determines when to end the process.
	while len(leaves)<args.size:
		if len(current_species)==1:
			i=0
		else:
			i=np.random.choice(range(len(current_species)))
		a=current_species[i]
		a.gene_duplication()
		a.speciation() # event of speciation in which two new species diverge from the previous one.
		current_species=[x for x in sp_tree.nodes() if sp_tree.out_degree(x)==0] # updates the "leaves" in species tree.
		leaves=[x for x in seq_tree.nodes() if seq_tree.out_degree(x)==0] # updates the "leaves" in sequences tree.
	if len(orthologs)<args.n_ort:
		print('Warning: few sequences to have %d ortholog groups!'%args.n_ort)
	###====================================================================================================
	colection=AlignIO.MultipleSeqAlignment([SeqRecord(Seq(seq.sequence), id=str(seq)) for seq in sequences.colection])
	AlignIO.write(colection, open('%s_all_sequences.%s'%(args.out, args.msa_format), 'w'), args.msa_format)
	alignment = AlignIO.MultipleSeqAlignment(build_MSA(seq_tree, first_seq))
	AlignIO.write(alignment, open('%s_current_sequences.%s'%(args.out, args.msa_format), 'w'), args.msa_format)
	tree=Phylo.BaseTree.Tree(root=build_tree(seq_tree, first_seq), rooted=True)
	Phylo.write(tree, '%s_gene_tree.%s'%(args.out, args.tree_format), args.tree_format)
	cladogram=Phylo.BaseTree.Tree(root=build_tree(sp_tree, first_sp, 'specie'), rooted=True)
	Phylo.write(cladogram, '%s_species_cladogram.%s'%(args.out, args.tree_format), args.tree_format)
	###====================================================================================================
	ortholog_groups={}
	for i,x in enumerate(orthologs):
		ortholog_groups[i]=[]
		for y in leaves:
			if nx.has_path(seq_tree, x, y):
				path=nx.shortest_path(seq_tree, x, y)
				free_way=True
				j=1
				while free_way and j<len(path):
					if path[j] in orthologs:
						free_way=False
					j+=1
				if free_way:
					ortholog_groups[i].append(y)
	d={'Sequence':[], 'OG':[]}
	for i,x in sorted(ortholog_groups.items(), key=lambda k: k[0]):
		for y in x:
			d['Sequence'].append(y)
			d['OG'].append(og_label(i))
	df=DataFrame(d)
	df.to_csv('%s_ortholog_groups.csv'%args.out, index=False)
	if args.n_ort>1:
		for i in ortholog_groups.keys():
			alignment = AlignIO.MultipleSeqAlignment([SeqRecord(Seq(x.sequence), id=str(x)) for x in ortholog_groups[i]])
			AlignIO.write(alignment, open('%s_OG_%s.%s'%(args.out, og_label(i), args.msa_format), 'w'), args.msa_format)
	###====================================================================================================
	d={'MSA\nColumns':[]}
	for i in range(len(orthologs)):
		d['OG %s'%og_label(i)]=[]
	for i in range(args.length):
		d['MSA\nColumns'].append(i+1)
		for j,x in enumerate(orthologs):
			msa = np.array([list(k.sequence) for k in ortholog_groups[j]])
			aa_types = set(msa[:,i])
			freq = {a:0 for a in aa_types if a != '-'}
			for a in msa[:,i]:
				if a != '-':
					freq[a]+=1
			freq = {k:float(v)/float(len(msa)) for k,v in freq.items()}
			if len(freq) > 0:
				d['OG %s'%og_label(j)].append(str(x.sthereochemistry[i])+' ('+' '.join(['{0}:{1:.2f}%'.format(seq3(k),v*100) for k,v in sorted(freq.items(), key = lambda x: x[1], reverse = True)])+')')
			else:
				d['OG %s'%og_label(j)].append(x.sthereochemistry[i])
	# df=pd.DataFrame(d)
	df=DataFrame(d)
	df.to_csv('%s_sthereochemistry.csv' % args.out, index=False)	
###====================================================================================================
### END
###====================================================================================================

	