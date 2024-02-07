# Finetuning
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Instruct
import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

# Inference
from transformers import pipeline

# Analysis
import gzip
import pickle
import matplotlib.pyplot as plt

# Group
from nltk.translate.bleu_score import sentence_bleu

modules = ['about', 'accarea', 'access', 'address', 'addrtyp', 'advancedSearch', 'advatsta', 'airplane', 'alarmclk', 'alocitem', 'alocwork', 'analyitd', 'analypdc', 'appCopierEdit', 'appGeneratorEdit', 'appInheritorEdit', 'approvalTransactions', 'appsched', 'appsWHBrowser', 'appsWHModuleSynchronise', 'assetAccountBalanceList', 'assetAccountTxnList', 'assetsAnalysisList', 'associatl', 'atsetobj', 'attrcbag', 'attrform', 'attribut', 'attributeValueEdit', 'attrilst', 'attrisat', 'attrisit', 'attrnode', 'attrslot', 'auditing', 'autopcal', 'autoplst', 'balanbus', 'balancos', 'balanfac', 'bankaedt', 'bankcode', 'batcerr', 'billcond', 'billing', 'billofma', 'billsing', 'blockers', 'budofedt', 'busiseg', 'busiyear', 'calendar', 'canceltxn', 'capacityPlanning', 'car', 'carbonco', 'CarPrio', 'cashDeposit', 'cashdisc', 'ccacbals', 'ccacbstr', 'chargbas', 'cheqregi', 'clipboard', 'clsstree', 'cmacbals', 'coacstat', 'cobjrept', 'columvar', 'comment', 'commiss', 'condaccn', 'condcond', 'condiset', 'condtedc', 'condtion', 'consult', 'coobwizz', 'corporateGroupEdit', 'costbook', 'costcent', 'costiobj', 'costmobj', 'costObjectiveBalanceList', 'costosel', 'costpobj', 'costsobj', 'costtype', 'cp_1252', 'cracbals', 'ctacbals', 'ctacbstr', 'curracc', 'currencyExchange', 'customagr', 'cxAccessNode', 'cxAccessoryList', 'cxApiKey', 'cxApplication', 'cxAsset', 'cxCampaign', 'cxCity', 'cxCombinedNomenclature', 'cxConditionedBag', 'cxContact', 'cxCounter', 'cxCreditCardAccount', 'cxCurrencyTable', 'cxCyberClient', 'cxDataConnector', 'cxDataField', 'cxDocument', 'cxDocumentComponent', 'cxDocumentIndex', 'cxGeneralLedger', 'cxIndustrialPlant', 'cxInstallationCharges', 'cxIntermediateStocking', 'cxIpAddress', 'cxItemDemand', 'cxModuleSettings', 'cxNeuralNetwork', 'cxPattItemNumber', 'cxPhrase', 'cxProceedings', 'cxProductionDataCapture', 'cxPurchaseReturn', 'cxReport', 'cxSalesProductConfiguration', 'cxSapBusinessOneStock', 'cxSignification', 'cxStateMonitor', 'cxStateStructure', 'cxStockSpace', 'cxStructure', 'cxTextCondition', 'cxTxnDescriptor', 'cxWebService', 'cxWidget', 'cxWorkflow', 'cxWorkFlowRoute', 'cxWorkTimeEvent', 'cxWorkTimeModel', 'cxWorkTimeRule', 'cxWorkTimeYear', 'cyber', 'databaseManage', 'dataConnectorImport', 'dataConnectorWebBrowser', 'dbaseviw', 'deacbals', 'defslot', 'deftrans', 'deliconf', 'delidisp', 'deliisel', 'delinote', 'deliveryNoteItemList', 'dialogue', 'directShipmentItem', 'dirshipm', 'dispobom', 'dnpycus', 'dnpysup', 'domadecl', 'dprcbook', 'dtausedt', 'dtazvedt', 'dunnsele', 'ecvatsta', 'eMailSentStatus', 'exacbals', 'excelcel', 'excelReader', 'eximport', 'ExpandNum', 'extdispo', 'family', 'favourit', 'fiacstat', 'finabook', 'finacopy', 'finajour', 'financialLoggingsList', 'finpcacc', 'finstand', 'flextime', 'floatfil', 'forecast', 'formula', 'fwdfabal', 'gantitdm', 'gantosup', 'gantt', 'gantwork', 'geledcla', 'geleddep', 'generalLedgerBalancesStructureList', 'genmodul', 'glacbals', 'graphicalQueryWizard', 'helpgen', 'holdinre', 'icastedt', 'ImportStockSpace', 'indexmgr', 'initsdat', 'inprovis', 'instchbook', 'intrastat', 'intrastat', 'invanaly', 'invcontr', 'inventoryAnalysis', 'inventoryCheck', 'inventoryFrequencyDistribution', 'inventoryImport', 'inventoryStratification', 'inventoryTaking', 'inventry', 'invoiitm', 'invoimaint', 'item', 'item', 'itemAlloc', 'itemDispositionEdit', 'itemDispositionLoggingsSelect', 'itemsea', 'itemVarianceAnalyze', 'jobRecordByDayWin', 'jobrecrd', 'jobsched', 'jobscond', 'jobssing', 'keyshtct', 'kpiAuditor', 'kpiMonitor', 'kpisupmo', 'language', 'legalPersonDeduplicate', 'legalPersonDuplicatesList', 'legalPersonNamesList', 'listvcol', 'literalAppsWH', 'literalSystem', 'loadeddlls', 'localeEdit', 'logcyber', 'logcyuse', 'loggiadm', 'loggibom', 'loggicid', 'loggicos', 'loggidac', 'loggidit', 'loggings', 'loggiocm', 'loggiocr', 'loggiode', 'loggioex', 'loggipic', 'loggipit', 'loggipoi', 'loggiprov', 'loggirit', 'loggisto', 'loggiwip', 'login', 'loginAgent', 'loginAlias', 'logout', 'loguser', 'machine', 'maintenc', 'maintpaym', 'masttool', 'member', 'metaaccs', 'metainfo', 'metamodl', 'metaobj', 'miniwb', 'missingAttributes', 'mt940edt', 'neuralNetworkLoad', 'neuralNetworkQuery', 'newsletter2Go', 'objcount', 'objctsql', 'objectStructureEdit', 'objectWebCrawler', 'objinsp', 'objnavi', 'odocu', 'offorder', 'offorder', 'offorder', 'offorita', 'offoritt', 'oitemsel', 'oitmsupp', 'olapCustomerList', 'olapRepresentativeList', 'olapSupplierList', 'olsync', 'onewaypa', 'openitem', 'openItemTxnSelect', 'openlist', 'opitcmac', 'opitcrac', 'opitdbac', 'opitdunn', 'opitexac', 'opitglac', 'orderAllocationResolve', 'orderfav', 'orderfav', 'ordergrp', 'ordermaint', 'orderPaymentStatisticsList', 'orderqui', 'orderrisk', 'orderStateStatisticsList', 'ordertxt', 'organizationChart', 'origpart', 'ortodnin', 'outlook', 'outprovis', 'packaccn', 'packitem', 'packload', 'parquery', 'partnerCastEdit', 'partpaym', 'password', 'paychequ', 'paydtaus', 'paydtazv', 'payevdnc', 'paymprop', 'paymt101', 'paysepa', 'performa', 'person', 'personDeduplicate', 'personDuplicatesList', 'personNamesList', 'pinvbook', 'plusbutton', 'pmedia', 'ppcrbals', 'ppdebals', 'presentationManagerEdit', 'preview', 'pricecal', 'PriceDiscount', 'PriceDiscountTable', 'prichap', 'prichas', 'print', 'printer', 'printtyp', 'prjgen', 'prjnmoni', 'processes_advancedemand', 'processes_attribute', 'processes_qm_bat', 'procfold', 'prodserv', 'product', 'Profiling', 'projaloc', 'projcost', 'projectGeneratorEdit', 'projinfo', 'proorder', 'proquery', 'provsing', 'pstgroup', 'purcappr', 'purccomp', 'purcdunn', 'purchaseInquiry', 'purchaseItem', 'purchaseOrder', 'purchaseOrderSignature', 'purchaseProposalsList', 'purchaseRequisition', 'purchaseRequisitionLoggingsList', 'purchaseService', 'purchaseSet', 'purchcre', 'purchinv', 'purchinvitem', 'purchinvlog', 'puriauto', 'puridunn', 'puriitem', 'purogedt', 'puroitem', 'purqitem', 'pusaitem', 'Pythia_cxAntiTerrorScreening', 'Pythia_sanctionsListMatch', 'Pythia_sanctionsListMonitor', 'Pythia_sanctionsListQuery', 'Pythia_xmlimprt_py', 'qm_alert_qm', 'qm_arithmetic_qm', 'qm_asciiFile_qm', 'qm_button_qm', 'qm_condBag_qm', 'qm_condWrapper_qm', 'qm_date_qm', 'qm_deadlock_qm', 'qm_dictionary_qm', 'qm_dragndrop_qm', 'qm_font_qm', 'qm_formula_qm', 'qm_garbage_qm', 'qm_grpwid_qm', 'qm_itempattern_qm', 'qm_link_qm', 'qm_listview_qm', 'qm_listviewAutoPos_qm', 'qm_listviewExceptions_qm', 'qm_listviewOboxEdit_qm', 'qm_listviewOboxUpDown_qm', 'qm_listviewOboxUpDown2_qm', 'qm_listviewSetFormat_qm', 'qm_listviewSort_qm', 'qm_listviewxml_qm', 'qm_message_qm', 'qm_mlole_qm', 'qm_olectl_qm', 'qm_patternQuery_qm', 'qm_periodicDate_qm', 'qm_phone_qm', 'qm_picture_qm', 'qm_pobox_qm', 'qm_printing_qm', 'qm_query_qm', 'qm_queryExist_qm', 'qm_rates_qm', 'qm_refbasedptr_qm', 'qm_reload_qm', 'qm_resume_qm', 'qm_rounding_qm', 'qm_security_qm', 'qm_setLocale_qm', 'qm_simplwid_qm', 'qm_spanDate_qm', 'qm_spanTime_qm', 'qm_systemObject_qm', 'qm_telephony_qm', 'qm_term_qm', 'qm_time_qm', 'qm_timedTrigger_qm', 'qm_tmprture_qm', 'qm_txnByCond_qm', 'qm_unit_qm', 'qm_unittestDLL_qm', 'qm_unittestIV_qm', 'qm_vector_qm', 'qm_wrapper_qm', 'qualassu', 'query', 'queryatt', 'queryWizard', 'receitem', 'receivingItemStatusList', 'receving', 'registerStorageAids', 'remotmsg', 'repclass', 'reporting', 'request', 'resolbom', 'resoljob', 'resset', 'restrict', 'routePlanningEdit', 'sacoest', 'salebase', 'salecedt', 'salecond', 'saleitem', 'saleset', 'salesItemOrderStatisticsList', 'salesOrderEngineering', 'salexitm', 'sapBusinessOneInterfaceMonitor', 'sarestat', 'saretour', 'scanner_login_app_scanner', 'scanner_main_app_scanner', 'scanner_main_info_iteminfo_app_scanner', 'scanner_main_info_queryitem_app_scanner', 'scanner_main_info_querystorage_app_scanner', 'scanner_main_info_storageinfo_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventorydown_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventoryup_app_scanner', 'scanner_main_maintenance_adjustinventory_app_scanner', 'scanner_main_maintenance_adjustinventory_changestatus_app_scanner', 'scanner_main_maintenance_relocate_app_scanner', 'scanner_main_maintenance_relocate_license_app_scanner', 'scanner_main_maintenance_stocktaking_app_scanner', 'scanner_main_maintenance_stocktaking_cyclecountstorage_app_scanner', 'scanner_main_print_labelorreport_app_scanner', 'scanner_main_print_printdocument_app_scanner', 'scanner_main_print_printitemlabel_app_scanner', 'scanner_main_print_printstoragelabel_app_scanner', 'scanner_main_processes_inbound_app_scanner', 'scanner_main_processes_inbound_directputaway_app_scanner', 'scanner_main_processes_inbound_receivefromcustomer_app_scanner', 'scanner_main_processes_inbound_receivefromsupplier_app_scanner', 'scanner_main_processes_outbound_app_scanner', 'scanner_main_processes_outbound_pick_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidatelicense_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidatelooseitems_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_identifylicensesofshipment_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pickandcollect_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pickanddrop_app_scanner', 'scanner_main_processes_transport_putaway_app_scanner', 'scanner_main_processes_transport_putaway_looseitemsputaway_app_scanner', 'scanner_select_forktruck_app_scanner', 'scanner_select_picklist_nui_app_scanner', 'scanner_select_pickzone_app_scanner', 'scanner_select_status_app_scanner', 'scanner_select_storage_app_scanner', 'scanner_select_workzone_app_scanner', 'scanner_show_consolidationstoragesstatus_app_scanner', 'schedule', 'secclass', 'secgroup', 'secmessg', 'secobjec', 'secsystm', 'serinumb', 'serviitt', 'servinqu', 'sessiond', 'setalloc', 'setlimit', 'setlocal', 'showmoni', 'showwrkf', 'sinvbook', 'slotsbas', 'spardire', 'sparitem', 'specifier', 'sstgroup', 'staffmem', 'startset', 'statfoca', 'statinst', 'statistx', 'statofit', 'statoitm', 'statordr', 'statpodc', 'statprpl', 'statturn', 'statwprg', 'statwrap', 'stoaccnt', 'stock', 'stockInput', 'stockOrder', 'stockOrder', 'stockSequentialTest', 'stockSpaceQuery', 'StockStatistics', 'stockSwitching', 'stocktxn', 'stockWithdrawal', 'stomobil', 'stotrans', 'substock', 'supplierAgreement', 'supplierItemList', 'synchrDB', 'sysnote', 'tapi', 'task', 'taxrate', 'telecomEdit', 'telecrep', 'testAllocation', 'testattr', 'testform', 'timeoffc', 'tool', 'truck', 'txnhisto', 'unitbill', 'unitCalculator', 'units', 'unittabl', 'updFClip', 'user', 'userhier', 'utilaccn', 'utilitem', 'utilofor', 'utilpart', 'utilpurc', 'vacaopen', 'validity', 'vatreturn', 'vehicle', 'warehouseMonitor', 'warehouseMonitor', 'webservice', 'windows', 'wipAccount', 'workarea', 'workflowGraphList', 'workgrup', 'workingTimeAccount', 'workstat', 'workTimeFlexiCalculate', 'workTimeTerminal', 'worldClock', 'z4report', 'ZUGFeRD']

classificationDataFile = "classificationData.pkl.gz"
classificationDataFileDesc = "classificationDataDesc.pkl.gz"

def instruct(promptsPerClass=""):
    instructModel = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(instructModel)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        instructModel,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    prompts = []
    with open("data/en_articles_classification_int.json", encoding="utf-8") as f:
        data = json.load(f)
    for module in data:
        prompts.append((f"I will now provide you with a description of a module. Please generate {promptsPerClass if promptsPerClass else 3 + int(np.ceil(len(module['input'].split()) / 6))} prompts that query which module is responsible for some given functionality, where this functionality stems from the module description, and the prompts use various formulations to ask which module is being described.\n\nModule description: {module['input']}", module['label']))
    
    instructions = []
    for prompt, label in prompts:
        # https://huggingface.co/blog/llama2
        llamaPrompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]
        """

        inputs = tokenizer(llamaPrompt, return_tensors="pt").to('cuda')

        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=4096,
                top_p=1,
                temperature=0.01,
            )
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.split("[/INST]", 1)[1].strip()

        if label % 100 == 0:
            print(label)
            print(response)

        pattern = re.compile(r'\d+\.\s(.+?)(?:\n|$)')
        matches = pattern.findall(response)
        for match in matches:
            instructions.append({"input": match.strip("\""), "label": label})

    with open(f"data/en_articles_classification_instruct{promptsPerClass}.json", "w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, indent=4)

def instructUniform(samplesPerClass=80):
    with open("data/en_articles_classification_instruct.json", encoding="utf-8") as f:
        data = json.load(f)

    dataByLabel = {}
    for sample in data:
        label = sample['label']
        if label not in dataByLabel:
            dataByLabel[label] = []
        dataByLabel[label].append(sample)

    for label in dataByLabel:
        currentSamples = dataByLabel[label]
        while len(dataByLabel[label]) < samplesPerClass:
            dataByLabel[label].extend(currentSamples)

        if len(dataByLabel[label]) > samplesPerClass:
            dataByLabel[label] = dataByLabel[label][:samplesPerClass]

    with open(f"data/en_articles_classification_instruct{samplesPerClass}uniform.json", "w", encoding="utf-8") as f:
        json.dump([sample for label in dataByLabel for sample in dataByLabel[label]], f, ensure_ascii=False, indent=4)

def instructBullet():
    instructModel = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(instructModel)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    model = AutoModelForCausalLM.from_pretrained(
        instructModel,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    with open("data/en_articles_classification_int.json", encoding="utf-8") as f:
        data = json.load(f)
    
    bullets = {}
    for module in data:
        prompt = f"I will now provide you with the description of a module. Please generate a numbered list covering each relevant aspect of the module description.\n\nModule description: {module['input']}"
        # https://huggingface.co/blog/llama2
        llamaPrompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]"""

        inputs = tokenizer(llamaPrompt, return_tensors="pt").to('cuda')       

        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=4096,
                top_p=1,
                temperature=0.01,
            )
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.split("[/INST]", 1)[1].strip()

        if label % 100 == 0:
            print(label)
            print(response)

        pattern = re.compile(r'\d+\.\s(.+?)(?:\n|$)')
        matches = pattern.findall(response)
        bullets[label] = {"bullets": [match.strip("\"") for match in matches], "description": module['input']}

    with open(f"data/bullets.json", "w", encoding="utf-8") as f:
        json.dump(bullets, f, ensure_ascii=False, indent=4)

    instructions = []
    for label in bullets:
        moduleDescription = bullets[label]["description"]
        for bullet in bullets[label]["bullets"]:
            prompt = f"I will now provide you with the description of a module, followed by a relevant aspect of the module description. Please generate a prompt based on the provided aspect that queries which module is responsible for some given functionality, where this functionality stems from the module description, and the prompts use various formulations to ask which module is being described.\n\nModule description: {moduleDescription}\n\nAspect:\n{bullet}"
            llamaPrompt = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST]"""
            inputs = tokenizer(llamaPrompt, return_tensors="pt").to('cuda')       

            outputs = model.generate(
                **inputs, 
                generation_config=GenerationConfig(
                    do_sample=True,
                    max_new_tokens=4096,
                    top_p=1,
                    temperature=0.01,
                )
            )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = text.split("[/INST]", 1)[1].strip()
            instructions.append({"input": response, "label": label})

    with open(f"data/en_articles_classification_instructBullet.json", "w", encoding="utf-8") as f:
        json.dump(instructions, f, ensure_ascii=False, indent=4)

def group():
    with open("data/en_articles_classification.json", encoding="utf-8") as f:
        data = json.load(f)

    indices = np.triu_indices(len(data), k=1)

    print(len(indices[0]))

    related = []
    for i, j in zip(*indices):
        if sentence_bleu([data[i]['input']], data[j]['input']) > 0.8:
            related.append((data[i]['label'], data[j]['label']))

    #print(related)

    def merge_tuples(tuple_list):
        connected_elements = {}

        # Function to find the root element in the connected set
        def find_root(element):
            if connected_elements[element] == element:
                return element
            return find_root(connected_elements[element])

        # Union operation to connect two elements
        def union(a, b):
            root_a = find_root(a)
            root_b = find_root(b)
            connected_elements[root_a] = root_b

        # Populate connected_elements dictionary
        for tuple_pair in tuple_list:
            a, b = tuple_pair
            if a not in connected_elements:
                connected_elements[a] = a
            if b not in connected_elements:
                connected_elements[b] = b
            union(a, b)

        # Create a dictionary to store connected sets
        connected_sets = {}
        for element in connected_elements:
            root = find_root(element)
            if root not in connected_sets:
                connected_sets[root] = [element]
            else:
                connected_sets[root].append(element)

        # Convert connected_sets values to lists
        result = list(connected_sets.values())

        return result

    return merge_tuples(related)

def finetuneNoEval(model_checkpoint, promptsPerClass=""):
    if not os.path.exists(f"data/en_articles_classification_instruct{promptsPerClass}.json"):
        with open(f"data/en_articles_classification_instruct10.json", encoding="utf-8") as f:
            data = json.load(f)
        
        dataByLabel = {}
        for sample in data:
            label = sample['label']
            if label not in dataByLabel:
                dataByLabel[label] = []
            dataByLabel[label].append(sample)
        
        truncatedData = []
        for label in dataByLabel:
            truncatedData.extend({"input": sample['input'], "label": sample['label']} for sample in dataByLabel[label][:promptsPerClass])
        with open(f"data/en_articles_classification_instruct{promptsPerClass}.json", "w", encoding="utf-8") as f:
            json.dump(truncatedData, f, ensure_ascii=False, indent=4)
    dataset = Dataset.from_json(f'data/en_articles_classification_instruct{promptsPerClass}.json')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["input"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=630)

    args = TrainingArguments(
        f"output/{model_checkpoint}-classification-instruct{promptsPerClass}",
        save_strategy = "steps",
        save_steps=20000,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        max_steps=200000,
        weight_decay=0.01,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "prf": precision_recall_fscore_support(labels, predictions, average="macro"),
        }

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    #trainer.train("output/bert-base-uncased-classification-instruct/checkpoint-140000")
    trainer.train()

def inference(modelName, threshold=None):
    prompts = [
        ("Which module provides version and copyright information?", 0),

        ("How can I calculate the current time in another location?", 627), # 627
        ("How can I calculate the current time in another location while accounting for discrepancies due to time zones?", 627), # 627
        ("With which module can I calculate the current time in another location while accounting for discrepancies due to time zones?", 627), # 627

        ("Tell me how to test the conversion of a temperature into the different heat units.", 460), # 460
        ("Where do I record both flexitime and operating data (BDE)?", 626), # 626
        ("Where can I check offer/order data?", 608), # 608
        ("Help me with inspection of partner data.", 609), # 609
        ("Provide me with resources on inspection of purchasing data.", 610), # 610

        ("Parts lists describe the composition of a production part. A bill of material consists of parts, which in turn can have a bill of material.", 45), # 45
        ("Which module am I referencing? Parts lists describe the composition of a production part. A bill of material consists of parts, which in turn can have a bill of material.", 45), # 45
    ]

    model = AutoModelForSequenceClassification.from_pretrained(f"output/{modelName}")
    tokenizer = AutoTokenizer.from_pretrained(re.search(r'/(.*?)-classification', modelName).group(1))

    if not threshold:
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

        for prompt in prompts:
            output = classifier(prompt[0])[0]
            label = int(output['label'].replace('LABEL_', ''))
            print(f"{modules[label]} ({label}):\t{output['score']}", modules[prompt[1]], prompt[1])
    else:
        for prompt in prompts:
            inputs = tokenizer(prompt[0], return_tensors="pt")
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            high_prob_indices = torch.where(probabilities > threshold)[1]
            high_probs = probabilities[0, high_prob_indices]
            labels = sorted(zip(high_prob_indices, high_probs), key=lambda x: x[1], reverse=True)

            print(prompt[0], "Correct module:", modules[prompt[1]], prompt[1])
            for label in labels:
                print(f"{modules[label[0]]}:\t{label[1]}")
            print()
            #print(outputs.logits.argmax(-1))

def analysis(modelName, specs, samplesFromEnd=2):
    with open("data/en_articles_classification_instruct10.json", encoding="utf-8") as f:
        data = json.load(f)

    dataByLabel = {}
    for sample in data:
        label = sample['label']
        if label not in dataByLabel:
            dataByLabel[label] = []
        dataByLabel[label].append(sample)

    testData = {label: dataByLabel[label][-samplesFromEnd:] for label in dataByLabel}

    if os.path.exists(classificationDataFile):
        with gzip.open(classificationDataFile, "rb") as f:
            classificationData = pickle.load(f)
    else:
        classificationData = {}
    
    change = False
    for spec in specs:
        currentModel = modelName.format(*spec)
        if currentModel in classificationData:
            continue
        else:
            change = True
            model = AutoModelForSequenceClassification.from_pretrained(f"output/{currentModel}")
            tokenizer = AutoTokenizer.from_pretrained(f"{spec[0]}bert-base-uncased")

            correctPrediction = 0
            confidences = []
            responses = []
            for label in testData:
                for sample in testData[label]:
                    inputs = tokenizer(sample['input'], return_tensors="pt")
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten().detach().cpu().numpy()
                    labels = sorted(zip(range(len(dataByLabel)), probabilities), key=lambda x: x[1], reverse=True)
                    responses.append(labels)
                    if label == labels[0][0]:
                        correctPrediction += 1
                    for labelTuple in labels:
                        if labelTuple[0] == label:
                            confidences.append(labelTuple[1])
                            break
            
            classificationData[currentModel] = {}
            classificationData[currentModel]['correctPredictions'] = (correctPrediction, len(dataByLabel) * samplesFromEnd)
            classificationData[currentModel]['confidence'] = confidences
            classificationData[currentModel]['response'] = responses
    if change:
        with gzip.open(classificationDataFile, "wb") as f:
            pickle.dump(classificationData, f)

    print("Correct predictions:", [classificationData[modelName.format(*spec)]['correctPredictions'] for spec in specs])
    print("Average confidence:", [np.average(classificationData[modelName.format(*spec)]['confidence']) for spec in specs])
    if (comp1 := ("distil", 5, 50000)) in specs and (comp2 := ("distil", 8, 50000)) in specs:
        print("Correlation between 5 and 8 samples per module (distilbert):", {np.corrcoef(classificationData[modelName.format(*comp1)]['confidence'], classificationData[modelName.format(*comp2)]['confidence'])[0, 1]})
    for spec in specs:
        currentModel = modelName.format(*spec)
        values, bins = np.histogram(classificationData[currentModel]['confidence'], bins=100)

        cumulative = np.cumsum(values)

        plt.plot(bins[:-1], cumulative, label=f"{spec[0]}bert-s{spec[1]}-c{spec[2]}: {classificationData[currentModel]['correctPredictions'][0]}, {np.average(classificationData[currentModel]['confidence']):.2f}")
    plt.legend()
    plt.xlabel("Confidence in correct module")
    plt.ylabel(f"Cumulative count (dataset size of {classificationData[modelName.format(*specs[0])]['correctPredictions'][1]} samples)")
    plt.title("Cumulative distribution of confidence")
    plt.show()
    plt.close()

    for spec in [("distil", 10, 50000), ("distil", 8, 50000), ("", 8, 120000), ("", "", 200000)]:
        if not spec in specs:
            continue
        print()

        currentModel = modelName.format(*spec)
        print(f"Worst performing samples ({spec[0]}bert, {spec[1]} samples per module):")
        worstSamples = sorted(zip([sample for label in testData for sample in testData[label]], classificationData[currentModel]['confidence'], classificationData[currentModel]['response']), key=lambda x: x[1])[:5]
        for sample in worstSamples:
            print(f"\t{sample[0]['input']}\n\t\tCorrect module: {modules[sample[0]['label']]} ({sample[1]:.2f})\n\t\tPredicted module: {modules[sample[2][0][0]]} ({sample[2][0][1]:.2f})")

def analysisDescription(modelName, specs):
    with open("data/en_articles_classification_instruct10.json", encoding="utf-8") as f:
        data = json.load(f)

    dataByLabel = {}
    for sample in data:
        label = sample['label']
        if label not in dataByLabel:
            dataByLabel[label] = []
        dataByLabel[label].append(sample)

    with open("data/en_articles_classification_int.json", encoding="utf-8") as f:
        data = json.load(f)

    testData = {sample["label"]: sample["input"] for sample in data}

    if os.path.exists(classificationDataFileDesc):
        with gzip.open(classificationDataFileDesc, "rb") as f:
            classificationData = pickle.load(f)
    else:
        classificationData = {}

    change = False
    for spec in specs:
        currentModel = modelName.format(*spec)
        if currentModel in classificationData:
            continue
        else:
            change = True
            model = AutoModelForSequenceClassification.from_pretrained(f"output/{currentModel}")
            tokenizer = AutoTokenizer.from_pretrained(f"{spec[0]}bert-base-uncased")

            correctPrediction = 0
            confidences = []
            responses = []

            for label in testData:
                inputs = tokenizer(testData[label], return_tensors="pt")
                if len(inputs['input_ids'][0]) > 512:
                    continue

                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten().detach().cpu().numpy()
                labels = sorted(zip(range(len(dataByLabel)), probabilities), key=lambda x: x[1], reverse=True)
                responses.append(labels)
                if label == labels[0][0]:
                    correctPrediction += 1
                for labelTuple in labels:
                    if labelTuple[0] == label:
                        confidences.append(labelTuple[1])
                        break

            classificationData[currentModel] = {}
            classificationData[currentModel]['correctPredictions'] = (correctPrediction, len(dataByLabel))
            classificationData[currentModel]['confidence'] = confidences
            classificationData[currentModel]['response'] = responses
    if change:
        with gzip.open(classificationDataFileDesc, "wb") as f:
            pickle.dump(classificationData, f)

    print("Correct predictions:", [classificationData[modelName.format(*spec)]['correctPredictions'] for spec in specs])
    print("Average confidence:", [np.average(classificationData[modelName.format(*spec)]['confidence']) for spec in specs])
    for spec in specs:
        currentModel = modelName.format(*spec)
        values, bins = np.histogram(classificationData[currentModel]['confidence'], bins=100)

        cumulative = np.cumsum(values)

        plt.plot(bins[:-1], cumulative, label=f"{spec[0]}bert-s{spec[1]}-c{spec[2]}: {classificationData[currentModel]['correctPredictions'][0]}, {np.average(classificationData[currentModel]['confidence']):.2f}")
    plt.legend()
    plt.xlabel("Confidence in correct module")
    plt.ylabel(f"Cumulative count (dataset size of {classificationData[modelName.format(*specs[0])]['correctPredictions'][1]} samples)")
    plt.title("Cumulative distribution of confidence")
    plt.show()
    plt.close()

if __name__ == '__main__':
    #instruct(10)
    #instruct()
    #instruct(80)
    #instructUniform()

    #relatedModules = group()
    #relatedModules = [['car', 'truck', 'vehicle'], ['cmacbals', 'cracbals', 'deacbals', 'exacbals', 'ppcrbals', 'ppdebals'], ['costmobj', 'costsobj'], ['cxSapBusinessOneStock', 'sapBusinessOneInterfaceMonitor'], ['dtausedt', 'dtazvedt'], ['jobRecordByDayWin', 'jobrecrd'], ['loggiocm', 'loggiocr', 'loggiode', 'loggioex'], ['loggiprov', 'loggirit'], ['member', 'specifier'], ['opitcrac', 'opitdbac'], ['paydtaus', 'paydtazv'], ['prichap', 'prichas'], ['qm_deadlock_qm', 'qm_resume_qm'], ['qm_listviewOboxUpDown_qm', 'qm_listviewOboxUpDown2_qm'], ['qm_spanTime_qm', 'qm_term_qm'], ['scanner_main_info_iteminfo_app_scanner', 'scanner_main_info_storageinfo_app_scanner'], ['scanner_main_maintenance_adjustinventory_adjustinventorydown_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventoryup_app_scanner'], ['utilpart', 'utilpurc']]
    #print(relatedModules)

    #finetuneNoEval("distilbert-base-uncased", 10)
    #finetuneNoEval("distilbert-base-uncased", 1)
    #finetuneNoEval("distilbert-base-uncased", 5)
    #finetuneNoEval("distilbert-base-uncased", 8)
    #finetuneNoEval("bert-base-uncased", 10)
    #finetuneNoEval("bert-base-uncased", 8)
    #finetuneNoEval("bert-base-uncased")
    #finetuneNoEval("distilbert-base-uncased", 80)
    #finetuneNoEval("distilbert-base-uncased", "80uniform")

    #inference("distilbert-base-uncased-classification/checkpoint-30000")
    #inference("distilbert-base-uncased-classification-noeval/checkpoint-30000")
    #inference("distilbert-base-uncased-classification-instruct/checkpoint-100000", 0.1)
    #inference("distilbert-base-uncased-classification-instruct10/checkpoint-100000")
    #inference("distilbert-base-uncased-classification-instruct10/checkpoint-100000", 0.1)

    #inference("../../bert.cpp/models/bert-base-uncased-classification-instruct10", 0.1)
    #inference("../../bert.cpp/models/bert-base-uncased-classification-instruct", 0.1)
    
    analysisDescription("{0}bert-base-uncased-classification-instruct{1}/checkpoint-{2}", [
        ("distil", 80, "180000"),
        ("distil", "80uniform", "180000"),
        ("", "", 140000),
        ("", 8, 100000),
        ("", 10, 100000),
        ])
    
    """
    with open("data/en_articles_classification_instruct.json", encoding="utf-8") as f:
        data = json.load(f)
    dataByLabel = {}
    for sample in data:
        label = sample['label']
        if label not in dataByLabel:
            dataByLabel[label] = []
        dataByLabel[label].append(sample)
    print(max(len(dataByLabel[label]) for label in dataByLabel), min(len(dataByLabel[label]) for label in dataByLabel))
    """

    exit()

    analysis("{0}bert-base-uncased-classification-instruct{1}/checkpoint-{2}", [
        ("distil", 1, 50000),
        ("distil", 5, 50000),
        ("distil", 8, 50000),
        ("distil", 8, 100000),
        ("distil", 10, 50000),
        ("", 8, 100000),
        ("", 8, 150000),
        ("", 10, 100000),
        ("", "", 140000),
        ("", "", 200000),
        ], 2)