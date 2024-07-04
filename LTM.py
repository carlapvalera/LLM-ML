from typing import TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

def get_similarity(query:str,vector_space:list, k:int) -> list[tuple[str,float]]: #TODO
    """
    Finds the k most relevant texts in a set of texts given a query text.
    
    Parameters:
        - query (str): The query text.
        - vector_space (list): A list of texts to search.
        - k (int): The number of most relevant texts to return.
    
    Returns:
        A list of tuples (text, score) where text is one of the k most 
        relevant texts and score is its degree of similarity to the query text.
    """

    if k==0 or len(vector_space)==0:
        return []
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Ajustar y transformar el texto de consulta y el conjunto de textos
    X = vectorizer.fit_transform([query] + vector_space)
    
    # Calcular la similitud de coseno entre el texto de consulta y cada texto en el conjunto
    similarities = cosine_similarity(X[0], X[1:])
    
    # Ordenar los textos por similitud de coseno en orden descendente
    sorted_indices = similarities.argsort()[0][::-1]
    
    k=min(k,len(sorted_indices))

    # Devolver los k textos más relevantes
    return [(vector_space[i], similarities[0][i]) for i in sorted_indices[:k]]


class LTM_Node:
    '''
        A node in the LTM network. 
        There is two kind of nodes:
            -The actual past conversations: 
                These kind of nodes cannot have any children
            -The summary nodes:
                This kind has any number of children in a dictionary (vector->node)
                His vector is just a summary of his children vectors
    '''

    def __init__(self, arg:str|list['LTM_Node'],is_leaf=True):

        if is_leaf:
            '''
                This represent a new memory in LTM
                conversation: the conversation that this node represents
                parents: a list of the nodes that are parents of this node
            '''
            self.vector:str = arg
            self.children: dict[str,'LTM_Node']|None = None
            self.is_leaf:bool = True
            self.parents:list['LTM_Node'] = []
            return

        '''
            This represent a summary of memories
            MemoryList: a list of the memories that this node summarizes (children nodes)
            parents: a list of the nodes that are parents of this node
        '''
        self.children: dict[str,'LTM_Node']|None = {x.vector:x for x in arg}
        self.vector:str = self.calculate_vector()
        self.is_leaf:bool=False
        self.parents=[]

    def insert(self,node:'LTM_Node'):
        '''
            Adds a new children. This node(self) cannot be a memory node
        '''
        if self.children==None:
            raise Exception("LTM Insertion Error")
        else:
            self.children[node.vector]=node
            node.parents.append(self)
        self.update()

    def mix_memories(self, vector:str, new_memory_node:'LTM_Node'):
        '''
            Mix a new memory with a child in a new summary node,
            The new summary will be son of this node instead of the mixed child 
            - vector: the vector of the child that will be mixed
            - new_memory_node: the new memory node
        '''
        old_child = self.children[vector]
        new_summary=LTM_Node([old_child,new_memory_node], False)
        new_memory_node.parents.append(new_summary)
        old_child.parents.remove(self)
        old_child.parents.append(new_summary)
        self.children.pop(vector)
        self.insert(new_summary)

    def update(self):
        '''
            Update the vector of this node and the parents of this node
        '''
        if self.children==None:
            return
        self.children={x.vector:x for _, x in self.children.items()}
        self.vector=self.calculate_vector()
        for p in self.parents:
            p.update()

    def get_space(self) -> list[str]|None:
        '''
            Return all the children vectors
        '''
        if self.is_leaf:
            return None
        return list(self.children.keys())
    
    def calculate_vector(self) -> str: #TODO
        '''
            Calculate the vector of this node based on his children
        '''
        pass

class LTM:
    '''
        This class represents the LTM of the model.
        Contains a LTM_Node graph starting with the root node.
        This structure has 2 hyper-parameters:
            - _lambda: the minimum level of relevance that a vector must have for a query
            - k_child: the number of vectors retrieved to a query
        The hyper parameter 'k_child' will also be relevant for the graph construction:
            - For k=1: The graph will be a tree
            - For k>1: The graph will be a DAG
    '''

    def __init__(self, _lambda:float=0.5, k_child:int=1):
        self.root=LTM_Node([], False)
        self._lambda=_lambda
        self.k_child=k_child
        self.where_to_insert=[] 
        #NOTE: THE VARIABLE 'where_to_insert' IS USED TO STORAGE A NEW MEMORY,
        # BASED ON THE SEARCH RESULTS HAS THE STRUCTURE [(node,vector)], WHERE:
        #       - node will be the father of the new memory
        #       - vector will be the most relevant vector

    def insert(self,new_memory:str):
        '''
            Insert a new memory in the LTM.
            
            It will be inserted on the nodes that were relevant for the prompt 
            stored in 'where_to_insert' variable updated in 'get_vector' method.
        '''
        new_node = LTM_Node(new_memory)

        for node, vector in self.where_to_insert:
            if node.vector==vector and not vector in node.children.keys():
                #NOTE: Situation 1: The most relevant node was a summary.
                node.insert(new_node)
            else:    
                #NOTE: Situation 2: The most relevant node was another memory.
                node.mix_memories(vector,new_node)
        
        self.where_to_insert=[]

    def get_vector(self, prompt:str) -> list[str]:
        '''
            Return the first k (from 'k_child' variable) most relevant vectors for the prompt
            
            Also, save the nodes that will be the parents of the new memory and the relevant vectors
            for new memory insertion.
        '''
        
        self.where_to_insert = [
            (node, vector) for node, vector, _ in 
            self.relevant_nodes(self.root, prompt,self._lambda)
        ]
        
        if len(self.where_to_insert)==0:
            #NOTE: CASE OF NO RELEVANT MEMORIES
            self.where_to_insert=[(self.root,self.root.vector)]
            return ['']
        
        return [vector for _,vector in self.where_to_insert]

    def relevant_nodes(self, node:LTM_Node, prompt:str,
        _lambda:float, solution=None) -> list[tuple[LTM_Node,str,float]]:
        '''
            Receives a LTM, a prompt and a lambda value
            Return a list of tuple (node, vector, similarity) where:
            - node is the node that gives the future parent vector of the memory
            - vector is the vector that was relevant for the prompt (greater than lambda)
            - similarity is the similarity between the vector and the prompt
            The tuple is also stored in solution
        '''
        if solution==None:
            solution=[]
        
        best_results = get_similarity(prompt, node.get_space(), self.k_child)
        
        #The following 3 list are made for a summary node. The i-est element represents each child
        child_node=[]
        child_threads=[]
        child_relevant_nodes=[]

        for vector, similarity in best_results:
            
            if node.children[vector].is_leaf:
                # On this case the future parent of the memory cannot be the vector itself.
                # So, we will return the parent of the vector and the memories will 
                # be mixed into a new summary that will be storage in the parent.
                if similarity>_lambda:
                    solution.append((node,vector, similarity))
                continue
            
            # If the relevant vector is a summary, we return the best result between this vector and
            # the recursive call on this vector. This process will be running on a thread
            child_node.append((node.children[vector],vector,similarity))
            child_relevant_nodes.append([])
            child_threads.append(threading.Thread(target=self.relevant_nodes, 
                args=(node.children[vector], prompt,max(_lambda,similarity), child_relevant_nodes[-1])))
            child_threads[-1].start()
        
        #Joining the threads
        for i in range(len(child_threads)):
            child_threads[i].join()
            _,_,similarity = child_node[i]
            if len(child_relevant_nodes[i])!=0:
                solution.append(child_relevant_nodes[i][0])
            elif similarity > _lambda:
                solution.append(child_node[i])
        
        solution.sort(key = lambda x:x[2], reverse=True)
        return solution