class Queue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, data):
        self.queue.append(data)
        print(f"Enqueued: {data}")
    
    def dequeue(self):
        if not self.is_empty():
            dequeued = self.queue.pop(0)
            print(f"Dequeued: {dequeued}")
            return dequeued
        else:
            print("Queue is empty. Cannot dequeue.")
            return None
    
    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            print("Queue is empty. No element to peek.")
            return None
    
    def is_empty(self):
        return len(self.queue) == 0
    
    def size(self):
        return len(self.queue)
    
    def display(self):
        if self.is_empty():
            print("Queue is empty.")
        else:
            print("Queue:", self.queue)


if __name__ == "__main__":
    q = Queue()
    
    q.enqueue(10)
    q.enqueue(20)
    q.enqueue(30)
    q.enqueue(40)
    
    q.display()  
    
    q.dequeue() 
    q.display()  
    
    front = q.peek()  
    print(f"Front element: {front}")
    
    print(f"Queue size: {q.size()}")  
    
    q.dequeue()
    q.dequeue()  
    q.dequeue()  
    q.dequeue()  
    q.display()  
