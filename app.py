import flask

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

app = flask.Flask(__name__)
app.debug = True

class PrefixMiddleware(object):
    def __init__(self, app, prefix=''):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]
        
app.wsgi_app = PrefixMiddleware(app.wsgi_app)

app.secret_key = "bacon"

modules = ['about', 'accarea', 'access', 'address', 'addrtyp', 'advancedSearch', 'advatsta', 'airplane', 'alarmclk', 'alocitem', 'alocwork', 'analyitd', 'analypdc', 'appCopierEdit', 'appGeneratorEdit', 'appInheritorEdit', 'approvalTransactions', 'appsched', 'appsWHBrowser', 'appsWHModuleSynchronise', 'assetAccountBalanceList', 'assetAccountTxnList', 'assetsAnalysisList', 'associatl', 'atsetobj', 'attrcbag', 'attrform', 'attribut', 'attributeValueEdit', 'attrilst', 'attrisat', 'attrisit', 'attrnode', 'attrslot', 'auditing', 'autopcal', 'autoplst', 'balanbus', 'balancos', 'balanfac', 'bankaedt', 'bankcode', 'batcerr', 'billcond', 'billing', 'billofma', 'billsing', 'blockers', 'budofedt', 'busiseg', 'busiyear', 'calendar', 'canceltxn', 'capacityPlanning', 'car', 'carbonco', 'CarPrio', 'cashDeposit', 'cashdisc', 'ccacbals', 'ccacbstr', 'chargbas', 'cheqregi', 'clipboard', 'clsstree', 'cmacbals', 'coacstat', 'cobjrept', 'columvar', 'comment', 'commiss', 'condaccn', 'condcond', 'condiset', 'condtedc', 'condtion', 'consult', 'coobwizz', 'corporateGroupEdit', 'costbook', 'costcent', 'costiobj', 'costmobj', 'costObjectiveBalanceList', 'costosel', 'costpobj', 'costsobj', 'costtype', 'cp_1252', 'cracbals', 'ctacbals', 'ctacbstr', 'curracc', 'currencyExchange', 'customagr', 'cxAccessNode', 'cxAccessoryList', 'cxApiKey', 'cxApplication', 'cxAsset', 'cxCampaign', 'cxCity', 'cxCombinedNomenclature', 'cxConditionedBag', 'cxContact', 'cxCounter', 'cxCreditCardAccount', 'cxCurrencyTable', 'cxCyberClient', 'cxDataConnector', 'cxDataField', 'cxDocument', 'cxDocumentComponent', 'cxDocumentIndex', 'cxGeneralLedger', 'cxIndustrialPlant', 'cxInstallationCharges', 'cxIntermediateStocking', 'cxIpAddress', 'cxItemDemand', 'cxModuleSettings', 'cxNeuralNetwork', 'cxPattItemNumber', 'cxPhrase', 'cxProceedings', 'cxProductionDataCapture', 'cxPurchaseReturn', 'cxReport', 'cxSalesProductConfiguration', 'cxSapBusinessOneStock', 'cxSignification', 'cxStateMonitor', 'cxStateStructure', 'cxStockSpace', 'cxStructure', 'cxTextCondition', 'cxTxnDescriptor', 'cxWebService', 'cxWidget', 'cxWorkflow', 'cxWorkFlowRoute', 'cxWorkTimeEvent', 'cxWorkTimeModel', 'cxWorkTimeRule', 'cxWorkTimeYear', 'cyber', 'databaseManage', 'dataConnectorImport', 'dataConnectorWebBrowser', 'dbaseviw', 'deacbals', 'defslot', 'deftrans', 'deliconf', 'delidisp', 'deliisel', 'delinote', 'deliveryNoteItemList', 'dialogue', 'directShipmentItem', 'dirshipm', 'dispobom', 'dnpycus', 'dnpysup', 'domadecl', 'dprcbook', 'dtausedt', 'dtazvedt', 'dunnsele', 'ecvatsta', 'eMailSentStatus', 'exacbals', 'excelcel', 'excelReader', 'eximport', 'ExpandNum', 'extdispo', 'family', 'favourit', 'fiacstat', 'finabook', 'finacopy', 'finajour', 'financialLoggingsList', 'finpcacc', 'finstand', 'flextime', 'floatfil', 'forecast', 'formula', 'fwdfabal', 'gantitdm', 'gantosup', 'gantt', 'gantwork', 'geledcla', 'geleddep', 'generalLedgerBalancesStructureList', 'genmodul', 'glacbals', 'graphicalQueryWizard', 'helpgen', 'holdinre', 'icastedt', 'ImportStockSpace', 'indexmgr', 'initsdat', 'inprovis', 'instchbook', 'intrastat', 'intrastat', 'invanaly', 'invcontr', 'inventoryAnalysis', 'inventoryCheck', 'inventoryFrequencyDistribution', 'inventoryImport', 'inventoryStratification', 'inventoryTaking', 'inventry', 'invoiitm', 'invoimaint', 'item', 'item', 'itemAlloc', 'itemDispositionEdit', 'itemDispositionLoggingsSelect', 'itemsea', 'itemVarianceAnalyze', 'jobRecordByDayWin', 'jobrecrd', 'jobsched', 'jobscond', 'jobssing', 'keyshtct', 'kpiAuditor', 'kpiMonitor', 'kpisupmo', 'language', 'legalPersonDeduplicate', 'legalPersonDuplicatesList', 'legalPersonNamesList', 'listvcol', 'literalAppsWH', 'literalSystem', 'loadeddlls', 'localeEdit', 'logcyber', 'logcyuse', 'loggiadm', 'loggibom', 'loggicid', 'loggicos', 'loggidac', 'loggidit', 'loggings', 'loggiocm', 'loggiocr', 'loggiode', 'loggioex', 'loggipic', 'loggipit', 'loggipoi', 'loggiprov', 'loggirit', 'loggisto', 'loggiwip', 'login', 'loginAgent', 'loginAlias', 'logout', 'loguser', 'machine', 'maintenc', 'maintpaym', 'masttool', 'member', 'metaaccs', 'metainfo', 'metamodl', 'metaobj', 'miniwb', 'missingAttributes', 'mt940edt', 'neuralNetworkLoad', 'neuralNetworkQuery', 'newsletter2Go', 'objcount', 'objctsql', 'objectStructureEdit', 'objectWebCrawler', 'objinsp', 'objnavi', 'odocu', 'offorder', 'offorder', 'offorder', 'offorita', 'offoritt', 'oitemsel', 'oitmsupp', 'olapCustomerList', 'olapRepresentativeList', 'olapSupplierList', 'olsync', 'onewaypa', 'openitem', 'openItemTxnSelect', 'openlist', 'opitcmac', 'opitcrac', 'opitdbac', 'opitdunn', 'opitexac', 'opitglac', 'orderAllocationResolve', 'orderfav', 'orderfav', 'ordergrp', 'ordermaint', 'orderPaymentStatisticsList', 'orderqui', 'orderrisk', 'orderStateStatisticsList', 'ordertxt', 'organizationChart', 'origpart', 'ortodnin', 'outlook', 'outprovis', 'packaccn', 'packitem', 'packload', 'parquery', 'partnerCastEdit', 'partpaym', 'password', 'paychequ', 'paydtaus', 'paydtazv', 'payevdnc', 'paymprop', 'paymt101', 'paysepa', 'performa', 'person', 'personDeduplicate', 'personDuplicatesList', 'personNamesList', 'pinvbook', 'plusbutton', 'pmedia', 'ppcrbals', 'ppdebals', 'presentationManagerEdit', 'preview', 'pricecal', 'PriceDiscount', 'PriceDiscountTable', 'prichap', 'prichas', 'print', 'printer', 'printtyp', 'prjgen', 'prjnmoni', 'processes_advancedemand', 'processes_attribute', 'processes_qm_bat', 'procfold', 'prodserv', 'product', 'Profiling', 'projaloc', 'projcost', 'projectGeneratorEdit', 'projinfo', 'proorder', 'proquery', 'provsing', 'pstgroup', 'purcappr', 'purccomp', 'purcdunn', 'purchaseInquiry', 'purchaseItem', 'purchaseOrder', 'purchaseOrderSignature', 'purchaseProposalsList', 'purchaseRequisition', 'purchaseRequisitionLoggingsList', 'purchaseService', 'purchaseSet', 'purchcre', 'purchinv', 'purchinvitem', 'purchinvlog', 'puriauto', 'puridunn', 'puriitem', 'purogedt', 'puroitem', 'purqitem', 'pusaitem', 'Pythia_cxAntiTerrorScreening', 'Pythia_sanctionsListMatch', 'Pythia_sanctionsListMonitor', 'Pythia_sanctionsListQuery', 'Pythia_xmlimprt_py', 'qm_alert_qm', 'qm_arithmetic_qm', 'qm_asciiFile_qm', 'qm_button_qm', 'qm_condBag_qm', 'qm_condWrapper_qm', 'qm_date_qm', 'qm_deadlock_qm', 'qm_dictionary_qm', 'qm_dragndrop_qm', 'qm_font_qm', 'qm_formula_qm', 'qm_garbage_qm', 'qm_grpwid_qm', 'qm_itempattern_qm', 'qm_link_qm', 'qm_listview_qm', 'qm_listviewAutoPos_qm', 'qm_listviewExceptions_qm', 'qm_listviewOboxEdit_qm', 'qm_listviewOboxUpDown_qm', 'qm_listviewOboxUpDown2_qm', 'qm_listviewSetFormat_qm', 'qm_listviewSort_qm', 'qm_listviewxml_qm', 'qm_message_qm', 'qm_mlole_qm', 'qm_olectl_qm', 'qm_patternQuery_qm', 'qm_periodicDate_qm', 'qm_phone_qm', 'qm_picture_qm', 'qm_pobox_qm', 'qm_printing_qm', 'qm_query_qm', 'qm_queryExist_qm', 'qm_rates_qm', 'qm_refbasedptr_qm', 'qm_reload_qm', 'qm_resume_qm', 'qm_rounding_qm', 'qm_security_qm', 'qm_setLocale_qm', 'qm_simplwid_qm', 'qm_spanDate_qm', 'qm_spanTime_qm', 'qm_systemObject_qm', 'qm_telephony_qm', 'qm_term_qm', 'qm_time_qm', 'qm_timedTrigger_qm', 'qm_tmprture_qm', 'qm_txnByCond_qm', 'qm_unit_qm', 'qm_unittestDLL_qm', 'qm_unittestIV_qm', 'qm_vector_qm', 'qm_wrapper_qm', 'qualassu', 'query', 'queryatt', 'queryWizard', 'receitem', 'receivingItemStatusList', 'receving', 'registerStorageAids', 'remotmsg', 'repclass', 'reporting', 'request', 'resolbom', 'resoljob', 'resset', 'restrict', 'routePlanningEdit', 'sacoest', 'salebase', 'salecedt', 'salecond', 'saleitem', 'saleset', 'salesItemOrderStatisticsList', 'salesOrderEngineering', 'salexitm', 'sapBusinessOneInterfaceMonitor', 'sarestat', 'saretour', 'scanner_login_app_scanner', 'scanner_main_app_scanner', 'scanner_main_info_iteminfo_app_scanner', 'scanner_main_info_queryitem_app_scanner', 'scanner_main_info_querystorage_app_scanner', 'scanner_main_info_storageinfo_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventorydown_app_scanner', 'scanner_main_maintenance_adjustinventory_adjustinventoryup_app_scanner', 'scanner_main_maintenance_adjustinventory_app_scanner', 'scanner_main_maintenance_adjustinventory_changestatus_app_scanner', 'scanner_main_maintenance_relocate_app_scanner', 'scanner_main_maintenance_relocate_license_app_scanner', 'scanner_main_maintenance_stocktaking_app_scanner', 'scanner_main_maintenance_stocktaking_cyclecountstorage_app_scanner', 'scanner_main_print_labelorreport_app_scanner', 'scanner_main_print_printdocument_app_scanner', 'scanner_main_print_printitemlabel_app_scanner', 'scanner_main_print_printstoragelabel_app_scanner', 'scanner_main_processes_inbound_app_scanner', 'scanner_main_processes_inbound_directputaway_app_scanner', 'scanner_main_processes_inbound_receivefromcustomer_app_scanner', 'scanner_main_processes_inbound_receivefromsupplier_app_scanner', 'scanner_main_processes_outbound_app_scanner', 'scanner_main_processes_outbound_pick_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidatelicense_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_consolidatelooseitems_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_identifylicensesofshipment_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pickandcollect_app_scanner', 'scanner_main_processes_outbound_shiptocustomer_pickanddrop_app_scanner', 'scanner_main_processes_transport_putaway_app_scanner', 'scanner_main_processes_transport_putaway_looseitemsputaway_app_scanner', 'scanner_select_forktruck_app_scanner', 'scanner_select_picklist_nui_app_scanner', 'scanner_select_pickzone_app_scanner', 'scanner_select_status_app_scanner', 'scanner_select_storage_app_scanner', 'scanner_select_workzone_app_scanner', 'scanner_show_consolidationstoragesstatus_app_scanner', 'schedule', 'secclass', 'secgroup', 'secmessg', 'secobjec', 'secsystm', 'serinumb', 'serviitt', 'servinqu', 'sessiond', 'setalloc', 'setlimit', 'setlocal', 'showmoni', 'showwrkf', 'sinvbook', 'slotsbas', 'spardire', 'sparitem', 'specifier', 'sstgroup', 'staffmem', 'startset', 'statfoca', 'statinst', 'statistx', 'statofit', 'statoitm', 'statordr', 'statpodc', 'statprpl', 'statturn', 'statwprg', 'statwrap', 'stoaccnt', 'stock', 'stockInput', 'stockOrder', 'stockOrder', 'stockSequentialTest', 'stockSpaceQuery', 'StockStatistics', 'stockSwitching', 'stocktxn', 'stockWithdrawal', 'stomobil', 'stotrans', 'substock', 'supplierAgreement', 'supplierItemList', 'synchrDB', 'sysnote', 'tapi', 'task', 'taxrate', 'telecomEdit', 'telecrep', 'testAllocation', 'testattr', 'testform', 'timeoffc', 'tool', 'truck', 'txnhisto', 'unitbill', 'unitCalculator', 'units', 'unittabl', 'updFClip', 'user', 'userhier', 'utilaccn', 'utilitem', 'utilofor', 'utilpart', 'utilpurc', 'vacaopen', 'validity', 'vatreturn', 'vehicle', 'warehouseMonitor', 'warehouseMonitor', 'webservice', 'windows', 'wipAccount', 'workarea', 'workflowGraphList', 'workgrup', 'workingTimeAccount', 'workstat', 'workTimeFlexiCalculate', 'workTimeTerminal', 'worldClock', 'z4report', 'ZUGFeRD']

classificationModel = AutoModelForSequenceClassification.from_pretrained(f"output/distilbert-base-uncased-classification")
classificationTokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
threshold = 0.1

generativeModel = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
generativeTokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
adapter_path = "output/adapter_model"
# Fixing some of the early LLaMA HF conversion issues.
generativeTokenizer.bos_token_id = 1
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
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
generativeModel = PeftModel.from_pretrained(base_model, adapter_path)
generativeModel.eval()

@app.route('/nlp_api?action=<string:req>', methods=['POST'])
def nlp_api(req):
    if req == "queryModuleName":
        prompt = flask.request.get_json()['prompt']

        inputs = classificationTokenizer(prompt[0], return_tensors="pt")
        outputs = classificationModel(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        high_prob_indices = torch.where(probabilities > 0.1)[1]
        high_probs = probabilities[0, high_prob_indices]
        labels = sorted(zip(high_prob_indices, high_probs), key=lambda x: x[1], reverse=True)

        response = {
            "modules": []
        }

        for label in labels:
            response["modules"].append({"moduleName": modules[label[0]], "confidence": label[1].item()})
        print(response)

        return flask.jsonify(response)
    
    if req == "queryModuleDescription":
        prompt = flask.request.get_json()['prompt']
        promptAlpaca = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\nResponse:"

        inputs = generativeTokenizer(promptAlpaca, return_tensors="pt").to('cuda')

        outputs = generativeModel.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=True,
                max_new_tokens=2000,
                top_p=1,
                temperature=0.01,
            )
        )

        text = generativeTokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.replace(promptAlpaca, '').strip()#

        return flask.jsonify({"moduleDescription": response})