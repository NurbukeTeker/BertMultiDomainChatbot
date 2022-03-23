# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:44:52 2022

@author: nurbuketeker
"""

import tensorflow as tf
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import LabelEncoder


device_name = tf.test.gpu_device_name()

device = torch.device("cpu")
print('GPU:', torch.cuda.get_device_name(0))

from transformers import AutoModelForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

labelencoder = LabelEncoder()

class_dictionary={0 :"ATIS",2:"ACID", 1: "Banking",3:"CLINC"}
 
def getModel(num):
    filename = ""
    dict_name= ""
    if num == 0:
        filename = "atis_model"
        dict_name=atis_dict
    elif num == 1:
        filename =  "banking_model"
        dict_name=bank_dict
    elif num == 2:
         filename =  "acid_model"
         dict_name=acid_dict
    elif num == 3:
        filename =  "clinc_model"
        dict_name=clinc_dict


    pytorch_model = AutoModelForSequenceClassification.from_pretrained(filename)
    return pytorch_model ,dict_name


def getModelPrediction(text,pytorch_model):
    test_texts_ = [text]
    
    input_ids = []
    attention_masks = []
    
    for text in test_texts_:
        encoded_dict = tokenizer.encode_plus(
                            text,                     
                            add_special_tokens = True, 
                            max_length = 20,          
                            pad_to_max_length = True,
                            return_attention_mask = True,  
                            return_tensors = 'pt',   
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
            
   
        
    test_labels_ = labelencoder.fit_transform( [1])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_labels_.tolist())
    
    batch_size = 32  
    
    prediction_data = TensorDataset(input_ids, attention_masks,labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    
    print('Prediction started on test data')
    pytorch_model.eval()
    predictions , true_labels = [], []
    
    
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
    
      with torch.no_grad():
          outputs = pytorch_model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
    
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to("cpu").numpy()
      
      predictions.append(logits)
      true_labels.append(label_ids)
    
    print('Prediction completed')
    
    prediction_set = []
    
    for i in range(len(true_labels)):
      pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
      prediction_set.append(pred_labels_i)
    
    prediction_scores = [item for sublist in prediction_set for item in sublist]
    return prediction_scores

def getDomainPrediction(text):
    pytorch_model = AutoModelForSequenceClassification.from_pretrained("my_multidomain_model")

    prediction_scores = getModelPrediction(text,pytorch_model)

    print(prediction_scores)
    prediction_class = class_dictionary[prediction_scores[0]]
    
    return prediction_scores[0], prediction_class
   
def getIntent(predicted_domain, text):
    pytorch_model,dict_name = getModel(predicted_domain)
    prediction_scores = getModelPrediction(text,pytorch_model)
    print(prediction_scores)    
    return dict_name[prediction_scores[0]]



atis_dict = {0: 'atis_abbreviation',
 1: 'atis_aircraft',
 2: 'atis_aircraft#atis_flight#atis_flight_no',
 3: 'atis_airfare',
 4: 'atis_airline',
 5: 'atis_airline#atis_flight_no',
 6: 'atis_airport',
 7: 'atis_capacity',
 8: 'atis_cheapest',
 9: 'atis_city',
 10: 'atis_distance',
 11: 'atis_flight',
 12: 'atis_flight#atis_airfare',
 13: 'atis_flight_no',
 14: 'atis_flight_time',
 15: 'atis_ground_fare',
 16: 'atis_ground_service',
 17: 'atis_ground_service#atis_ground_fare',
 18: 'atis_meal',
 19: 'atis_quantity',
 20: 'atis_restriction'}


bank_dict ={0: 'Refund_not_showing_up',
 1: 'activate_my_card',
 2: 'age_limit',
 3: 'apple_pay_or_google_pay',
 4: 'atm_support',
 5: 'automatic_top_up',
 6: 'balance_not_updated_after_bank_transfer',
 7: 'balance_not_updated_after_cheque_or_cash_deposit',
 8: 'beneficiary_not_allowed',
 9: 'cancel_transfer',
 10: 'card_about_to_expire',
 11: 'card_acceptance',
 12: 'card_arrival',
 13: 'card_delivery_estimate',
 14: 'card_linking',
 15: 'card_not_working',
 16: 'card_payment_fee_charged',
 17: 'card_payment_not_recognised',
 18: 'card_payment_wrong_exchange_rate',
 19: 'card_swallowed',
 20: 'cash_withdrawal_charge',
 21: 'cash_withdrawal_not_recognised',
 22: 'change_pin',
 23: 'compromised_card',
 24: 'contactless_not_working',
 25: 'country_support',
 26: 'declined_card_payment',
 27: 'declined_cash_withdrawal',
 28: 'declined_transfer',
 29: 'direct_debit_payment_not_recognised',
 30: 'disposable_card_limits',
 31: 'edit_personal_details',
 32: 'exchange_charge',
 33: 'exchange_rate',
 34: 'exchange_via_app',
 35: 'extra_charge_on_statement',
 36: 'failed_transfer',
 37: 'fiat_currency_support',
 38: 'get_disposable_virtual_card',
 39: 'get_physical_card',
 40: 'getting_spare_card',
 41: 'getting_virtual_card',
 42: 'lost_or_stolen_card',
 43: 'lost_or_stolen_phone',
 44: 'order_physical_card',
 45: 'passcode_forgotten',
 46: 'pending_card_payment',
 47: 'pending_cash_withdrawal',
 48: 'pending_top_up',
 49: 'pending_transfer',
 50: 'pin_blocked',
 51: 'receiving_money',
 52: 'request_refund',
 53: 'reverted_card_payment?',
 54: 'supported_cards_and_currencies',
 55: 'terminate_account',
 56: 'top_up_by_bank_transfer_charge',
 57: 'top_up_by_card_charge',
 58: 'top_up_by_cash_or_cheque',
 59: 'top_up_failed',
 60: 'top_up_limits',
 61: 'top_up_reverted',
 62: 'topping_up_by_card',
 63: 'transaction_charged_twice',
 64: 'transfer_fee_charged',
 65: 'transfer_into_account',
 66: 'transfer_not_received_by_recipient',
 67: 'transfer_timing',
 68: 'unable_to_verify_identity',
 69: 'verify_my_identity',
 70: 'verify_source_of_funds',
 71: 'verify_top_up',
 72: 'virtual_card_not_working',
 73: 'visa_or_mastercard',
 74: 'why_verify_identity',
 75: 'wrong_amount_of_cash_received',
 76: 'wrong_exchange_rate_for_cash_withdrawal'}

acid_dict ={0: 'INFO_ADD_HOUSE',
 1: 'INFO_ADD_REMOVE_INSURED',
 2: 'INFO_ADD_REMOVE_VEHICLE',
 3: 'INFO_ADD_VEHICLE_PROPERTY_PAPERLESS_BILLING',
 4: 'INFO_AGENT_WRONG',
 5: 'INFO_AGT_NOT_RESPONDING',
 6: 'INFO_AMERICAN_STAR',
 7: 'INFO_AMT_DUE',
 8: 'INFO_AST_PURCHASE',
 9: 'INFO_AST_QUOTE',
 10: 'INFO_ATV_INS_EXPLAN',
 11: 'INFO_AUTO_COV_QUESTION',
 12: 'INFO_AUTO_INS_CANADA',
 13: 'INFO_AUTO_POLICY_CANT_SEE_IN_ACCT',
 14: 'INFO_AUTO_PYMT_CANCEL',
 15: 'INFO_AUTO_PYMT_MIN_BALANCE',
 16: 'INFO_AUTO_PYMT_SCHEDULE',
 17: 'INFO_BILLING_ACCT_NAME_EDIT',
 18: 'INFO_BILLING_ACCT_NUM',
 19: 'INFO_BILLING_DEPT_CONTACT',
 20: 'INFO_BILL_DUE_DATE',
 21: 'INFO_BOAT_COV_EXPLAN',
 22: 'INFO_BUSINESS_POLICY_CANT_SEE',
 23: 'INFO_CANCEL_CONFIRM',
 24: 'INFO_CANCEL_FEE',
 25: 'INFO_CANCEL_INS_POLICY',
 26: 'INFO_CANT_SEE_FARM_RANCH_POLICY',
 27: 'INFO_CANT_SEE_POLICY',
 28: 'INFO_CAREERS',
 29: 'INFO_CFR_QUESTION_GENERAL',
 30: 'INFO_CHANGE_AGENT',
 31: 'INFO_CHANGE_AUTOPAY_DATE',
 32: 'INFO_CHANGE_BANK_ACCT',
 33: 'INFO_CHANGE_USERID',
 34: 'INFO_CL_ADJUSTER_INFO',
 35: 'INFO_CL_CHECK_STATUS',
 36: 'INFO_CL_CLAIM_FILED',
 37: 'INFO_CL_COMPLAINT',
 38: 'INFO_CL_DOCS_EMAIL',
 39: 'INFO_CL_DOCS_FAX',
 40: 'INFO_CL_DOCS_MAIL',
 41: 'INFO_CL_DOCS_SEND',
 42: 'INFO_CL_DRP_ASSIGN',
 43: 'INFO_CL_DRP_JOIN',
 44: 'INFO_CL_FILE_CLAIM',
 45: 'INFO_CL_FNOL',
 46: 'INFO_CL_FNOL_AUTO_HAIL',
 47: 'INFO_CL_GLASS_SAFELITE',
 48: 'INFO_CL_HRP_JOIN',
 49: 'INFO_CL_RENTAL',
 50: 'INFO_CL_SHOP_ADD_WORK',
 51: 'INFO_CL_SHOP_SEND_ESTIMATE',
 52: 'INFO_CL_STATUS',
 53: 'INFO_CL_UPDATE_INFO',
 54: 'INFO_COLLECTIONS',
 55: 'INFO_COLL_COV_EXPLAN',
 56: 'INFO_COMBINE_PYMTS',
 57: 'INFO_COMP_COV_EXPLAN',
 58: 'INFO_CONFIRM_COVERAGE',
 59: 'INFO_CREDIT_CARD_CHANGE_NUM',
 60: 'INFO_CREDIT_CARD_FEE',
 61: 'INFO_CUSTOMER_SERVICE_HOURS',
 62: 'INFO_DEC_PAGE_NEEDED',
 63: 'INFO_DEDUCTIBLE',
 64: 'INFO_DED_EXPLAN',
 65: 'INFO_DELETE_DUPE_PYMT',
 66: 'INFO_DIFFERENT_AMTS',
 67: 'INFO_DISCOUNTS',
 68: 'INFO_DO_NOT_CONTACT',
 69: 'INFO_DREAMKEEP_REWARDS',
 70: 'INFO_DREAMKEEP_REWARDS_ERRORS',
 71: 'INFO_DREAMS_FOUNDATION',
 72: 'INFO_EMPLOYMENT_VERIFY',
 73: 'INFO_ERS',
 74: 'INFO_ERS_CONTACT',
 75: 'INFO_ERS_REIMBURSE',
 76: 'INFO_FIND_AGENT',
 77: 'INFO_FLOOD_INS_EXPLAN',
 78: 'INFO_FORGOT_EMAIL',
 79: 'INFO_FORGOT_PASSWORD',
 80: 'INFO_FORGOT_USERID',
 81: 'INFO_GAP_COVERAGE',
 82: 'INFO_GEN_POLICY_COV_QUESTION',
 83: 'INFO_GET_A_QUOTE_AUTO',
 84: 'INFO_GET_A_QUOTE_AUTO_NONOWNER',
 85: 'INFO_GET_A_QUOTE_CFR',
 86: 'INFO_GET_A_QUOTE_OTHER',
 87: 'INFO_GET_A_QUOTE_RENTERS',
 88: 'INFO_GET_A_QUOTE_RENTERS_PURCHASE',
 89: 'INFO_GLASS_COV',
 90: 'INFO_HANDLING_FEE_REMOVE',
 91: 'INFO_HEALTH_INS_QUOTE',
 92: 'INFO_HOMESITE_CONTACT',
 93: 'INFO_INS_CARD_PRINT',
 94: 'INFO_INS_CARD_PROOF',
 95: 'INFO_INS_CARD_SEND',
 96: 'INFO_INS_NOT_AVAILABLE',
 97: 'INFO_KNOWYOURDRIVE',
 98: 'INFO_KNOWYOURDRIVE_DEVICE_ACTIVATE',
 99: 'INFO_KNOWYOURDRIVE_DEVICE_RETURN',
 100: 'INFO_KNOWYOURDRIVE_ERRORS',
 101: 'INFO_LETTER_OF_EXPERIENCE',
 102: 'INFO_LIAB_EXPLAN',
 103: 'INFO_LIFE_BENEFICIARY_CHANGE',
 104: 'INFO_LIFE_CASH_OUT',
 105: 'INFO_LIFE_INCR_COV',
 106: 'INFO_LIFE_POLICY_AMT_DUE',
 107: 'INFO_LIFE_POLICY_AUTO_PYMT',
 108: 'INFO_LIFE_POLICY_CANCEL',
 109: 'INFO_LIFE_POLICY_CANNOT_SEE',
 110: 'INFO_LIFE_QUESTION_GENERAL',
 111: 'INFO_LIFE_REFUND',
 112: 'INFO_LIFE_UPDATE_CONTACT_INFO',
 113: 'INFO_LOGIN_ERROR',
 114: 'INFO_LOG_OUT',
 115: 'INFO_MAIL_PYMT_ADDRESS',
 116: 'INFO_MAKE_PYMT',
 117: 'INFO_MEXICO_AUTO_INS',
 118: 'INFO_MORTGAGE_CO_POI',
 119: 'INFO_NAME_CHANGE',
 120: 'INFO_NEW_VEHICLE_GRACE_PERIOD',
 121: 'INFO_ONE_TIME_PYMT',
 122: 'INFO_OPERATING_AREA',
 123: 'INFO_OPERATING_CO',
 124: 'INFO_PAPERLESS_DOCS_SETUP',
 125: 'INFO_PAPERLESS_DOCS_STOP',
 126: 'INFO_PAPERLESS_MAIL',
 127: 'INFO_PAY_LIFE_INS',
 128: 'INFO_PHONE_NUM',
 129: 'INFO_PHONE_NUM_INTERNATIONAL',
 130: 'INFO_POI_OLD',
 131: 'INFO_POLICY_DOC_NEEDED',
 132: 'INFO_POLICY_NUM',
 133: 'INFO_POLICY_TRANS_TO_RENTAL',
 134: 'INFO_PREMIUM_BREAKDOWN',
 135: 'INFO_PREPAID_CARD_PYMT',
 136: 'INFO_PROFILE_SECTION',
 137: 'INFO_PYMT_CONFIRM',
 138: 'INFO_PYMT_DUEDATE_CHANGE',
 139: 'INFO_PYMT_ERROR',
 140: 'INFO_PYMT_HISTORY',
 141: 'INFO_PYMT_NOT_ONTIME',
 142: 'INFO_PYMT_PROCESS_CHANGE',
 143: 'INFO_PYMT_SETUP_AUTO_PYMT',
 144: 'INFO_PYMT_TIME',
 145: 'INFO_REFUND_CHECK',
 146: 'INFO_REINSTATE_INS_POLICY',
 147: 'INFO_RENTERS_COV_EXPLAN',
 148: 'INFO_RIDESHARE_COV',
 149: 'INFO_RV_INS_EXPLAN',
 150: 'INFO_SALVAGE_VEHICLE',
 151: 'INFO_SET_UP_ACCT',
 152: 'INFO_SPEAK_TO_REP',
 153: 'INFO_TEEN_SAFE_DRIVER_SIGNUP',
 154: 'INFO_THE_GENERAL_CONTACT',
 155: 'INFO_TRANSFER_ACCT_BALANCE',
 156: 'INFO_TRAVEL_INS_EXPLAN',
 157: 'INFO_UPDATE_CONTACT_INFO',
 158: 'INFO_UPDATE_EMAIL',
 159: 'INFO_UPDATE_LIENHOLDER',
 160: 'INFO_UPDATE_PHONE_NUM',
 161: 'INFO_UW_ALUMNI_DISCOUNT',
 162: 'INFO_WHO_IS_MY_AGENT',
 163: 'INFO_WHY_WAS_POLICY_CANCELLED',
 164: 'INFO_srtwentytwo',
 165: 'NO',
 166: 'ST_GENERAL_REQUEST',
 167: 'ST_HELLO',
 168: 'ST_HOW_IS_ABBY',
 169: 'ST_HOW_OLD_IS_ABBY',
 170: 'ST_IS_ABBY_REAL',
 171: 'ST_THANK_YOU',
 172: 'ST_WHAT_CAN_ABBY_DO',
 173: 'ST_WHERE_DOES_ABBY_LIVE',
 174: 'YES'}

clinc_dict ={0: 0,
 1: 1,
 2: 2,
 3: 3,
 4: 4,
 5: 5,
 6: 6,
 7: 7,
 8: 8,
 9: 9,
 10: 10,
 11: 11,
 12: 12,
 13: 13,
 14: 14,
 15: 15,
 16: 16,
 17: 17,
 18: 18,
 19: 19,
 20: 20,
 21: 21,
 22: 22,
 23: 23,
 24: 24,
 25: 25,
 26: 26,
 27: 27,
 28: 28,
 29: 29,
 30: 30,
 31: 31,
 32: 32,
 33: 33,
 34: 34,
 35: 35,
 36: 36,
 37: 37,
 38: 38,
 39: 39,
 40: 40,
 41: 41,
 42: 42,
 43: 43,
 44: 44,
 45: 45,
 46: 46,
 47: 47,
 48: 48,
 49: 49,
 50: 50,
 51: 51,
 52: 52,
 53: 53,
 54: 54,
 55: 55,
 56: 56,
 57: 57,
 58: 58,
 59: 59,
 60: 60,
 61: 61,
 62: 62,
 63: 63,
 64: 64,
 65: 65,
 66: 66,
 67: 67,
 68: 68,
 69: 69,
 70: 70,
 71: 71,
 72: 72,
 73: 73,
 74: 74,
 75: 75,
 76: 76,
 77: 77,
 78: 78,
 79: 79,
 80: 80,
 81: 81,
 82: 82,
 83: 83,
 84: 84,
 85: 85,
 86: 86,
 87: 87,
 88: 88,
 89: 89,
 90: 90,
 91: 91,
 92: 92,
 93: 93,
 94: 94,
 95: 95,
 96: 96,
 97: 97,
 98: 98,
 99: 99,
 100: 100,
 101: 101,
 102: 102,
 103: 103,
 104: 104,
 105: 105,
 106: 106,
 107: 107,
 108: 108,
 109: 109,
 110: 110,
 111: 111,
 112: 112,
 113: 113,
 114: 114,
 115: 115,
 116: 116,
 117: 117,
 118: 118,
 119: 119,
 120: 120,
 121: 121,
 122: 122,
 123: 123,
 124: 124,
 125: 125,
 126: 126,
 127: 127,
 128: 128,
 129: 129,
 130: 130,
 131: 131,
 132: 132,
 133: 133,
 134: 134,
 135: 135,
 136: 136,
 137: 137,
 138: 138,
 139: 139,
 140: 140,
 141: 141,
 142: 142,
 143: 143,
 144: 144,
 145: 145,
 146: 146,
 147: 147,
 148: 148,
 149: 149,
 150: 150}
