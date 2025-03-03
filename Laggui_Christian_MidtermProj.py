!pip install apriori_python
from apriori_python.apriori import apriori
import time
import pandas as pd
from itertools import combinations, chain
from collections import defaultdict

# Store DataSet Directories
def openStore(value):
  value = int(value)
  match value:
    case 1:
      return './Data/Amazon'
    case 2:
      return './Data/BestBuy/'
    case 3:
      return './Data/Kmart/'
    case 4:
      return './Data/Nike/'
    case 5:
      return './Data/Staples/'

#function to count itemset frequency
def count_items(itemset, transactions):
  count = 0
  #iterate through the transactions to check if item is frequent
  for i in range(len(transactions)):
    #check if the item is in the transaction. increment count if true
    #if set(itemset).issubset(transactions.loc[i, 'Transaction']):
    if all(item in transactions.loc[i, 'Transaction'] for item in itemset):
      count += 1
  return count

#function to get unique items in transaction dataframe
def CreateItemset(transactions):
  allItems = transactions['Transaction'].str.split(', ').explode().apply(lambda x: x.strip()).unique()
  return allItems


def aprioriItemset(transactions, itemset, min_support, k = 1, output = {}, ):

  #k_output is the temporary dictionary that holds information for this iteration
  #   size:
  #   frequent_items: {}
  #   singleton_items: []

  k_output = {}
  k_output['size'] = k

  #temporary structures to hold values
  frequent_itemsets = {}
  pruned_itemsets = []
  singleton_items = []

  #iterate through each item- check if they are frequent
  for item in itemset:
    #after each item, check if the item is > min_support. Append to output if true
    frequentItem, support = checkSupport(transactions, item, min_support)
    if frequentItem:
      frequent_itemsets[f'{item}'] = support
      singleton_items = addSingleton(item, singleton_items)
    else:
      pruned_itemsets.append(item)

  #exit the function if there are no more combinations available.
  if len(frequent_itemsets.keys()) == 0:
    print(f'No frequent itemsets with size {k}')
    output['max_size'] = k-1
    return output
  #if we found frequent items in this iteration, try to iterate once more k++
  else:
    k_output['frequent_itemsets'] = frequent_itemsets
    #k_output['singleton_items'] = singleton_items
    output[f'{k}'] = k_output
    k += 1

  #recursively call the function
  aprioriItemset(transactions, getCombinations(singleton_items, k), min_support, k)
  return output

#function to calcualte the support of an itemset
def checkSupport(transactions, itemset, min_support):
  support = count_items(itemset, transactions) / len(transactions) * 100
  if support >= min_support:
    return True, support
  else:
    return False, support

def getCombinations(itemset, k):
  return list(combinations(itemset, k))

def addSingleton(item, singleton_list):
  singletonSet = set(singleton_list)
  # If the item is a tuple, iterate over its elements
  if isinstance(item, tuple):
      for sub_item in item:
          # Add the sub_item to the list if it's not already in the set
          if sub_item not in singleton_list:
              singleton_list.append(sub_item)
              singletonSet.add(sub_item)  # Keep the set updated
  # If the item is a single string, check if it's unique
  elif isinstance(item, str):
      if item not in singleton_list:
          singleton_list.append(item)
          singletonSet.add(item)  # Keep the set updated
  return singleton_list

def checkConfidence(input, min_confidence):
  k = input['max_size']
  output = {}
  while k > 1:
    itemsets = list(input[f'{k}']['frequent_itemsets'].keys())
    for i in itemsets:
      itemset = eval(i)
      for j in list(itemset):
        confidence = input[f'{k}']['frequent_itemsets'][i] / input['1']['frequent_itemsets'][j] * 100
        consequent = [i for i in itemset if i != j]
        if confidence >= float(min_confidence):
          output[f'{j} -> {", ".join(consequent)}'] = confidence
          if(k == input['max_size']):
            print(f'{j} -> {", ".join(consequent)}: \n Support: {input[f"{k}"]["frequent_itemsets"][i]} | Confidence: {confidence}')
    k -= 1
  return output


# Ask for input. Select the store
storeSelected = input("Which store would you like to explore?\n1. Amazon \n2. BestBuy \n3. Kmart \n4. Nike \n5. Staples \n\n")

path = openStore(storeSelected)

supportNum = input("\nmin_support %:\t")
confidenceNum = input("\nmin_confidence %:\t")
# Load and clean the datasets
with open(path + '/Transactions.csv', 'r') as transaction_file:
  transactions = pd.read_csv(transaction_file)
  try:
    with open(path + '/Items.csv', 'r') as item_file:
      itemset = pd.read_csv(item_file)
  except FileNotFoundError:
    print(f"File not found: {item_file}")
  else:
    CreateItemset(transactions)
  finally:
    #print(transactions)
    start_time = time.time()
    frequent_itemsets = aprioriItemset(transactions, itemset['Item Name'], float(supportNum))
    confidence = checkConfidence(frequent_itemsets, confidenceNum)
    for i, (key,value) in enumerate(confidence.items()):
      print(f'Rule {i+1}: {key}: {value}\n')
      
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    
    transactions['Transaction'] = transactions['Transaction'].apply(lambda x: x.split(', '))
    df_transactions = transactions.drop('Transaction ID', axis=1)
    #print(df_transactions)
    # Convert the DataFrame into a list of transactions (rows to lists)
    transactions = df_transactions.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
    #print(transactions)
    flattened_transactions = [transaction[0] for transaction in transactions]

    start_time = time.time()
    freqItemSet, rules = apriori(flattened_transactions, float(supportNum)/100, float(confidenceNum)/100)

    print("\n\nApriori:\n")
    for i, rule in enumerate(rules):
      print(f"Rule {i + 1}: {rule}\n")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
