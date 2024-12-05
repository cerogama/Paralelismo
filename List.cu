#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

struct Node {
    int data;
    Node* next;
};

// Versión en CPU de insertar un nodo al final
void insertAtEndCPU(Node** head, int data) {
    Node* newNode = new Node();
    newNode->data = data;
    newNode->next = nullptr;

    if (*head == nullptr) {
        *head = newNode;
    }
    else {
        Node* temp = *head;
        while (temp->next != nullptr) {
            temp = temp->next;
        }
        temp->next = newNode;
    }
}

// Versión en CPU de insertar un nodo en una posición específica
void insertAtPositionCPU(Node** head, int data, int position) {
    Node* newNode = new Node();
    newNode->data = data;

    if (position == 0) {
        newNode->next = *head;
        *head = newNode;
    }
    else {
        Node* temp = *head;
        int count = 0;
        while (temp != nullptr && count < position - 1) {
            temp = temp->next;
            count++;
        }

        if (temp != nullptr) {
            newNode->next = temp->next;
            temp->next = newNode;
        }
    }
}

// Versión en CPU de eliminar un nodo
void deleteNodeCPU(Node** head, int data) {
    if (*head == nullptr) return;

    Node* temp = *head;
    Node* prev = nullptr;

    if (temp != nullptr && temp->data == data) {
        *head = temp->next;
        delete temp;
        return;
    }

    while (temp != nullptr && temp->data != data) {
        prev = temp;
        temp = temp->next;
    }

    if (temp == nullptr) return;

    prev->next = temp->next;
    delete temp;
}

// Versión en CPU para imprimir la lista
void printListCPU(Node* head) {
    Node* current = head;
    while (current != nullptr) {
        std::cout << "Data: " << current->data << std::endl;
        current = current->next;
    }
}

// Versión en GPU de insertar un nodo al final
__global__ void insertAtEndGPU(Node** head, int data) {
    Node* newNode = (Node*)malloc(sizeof(Node));  // Asignación manual (no recomendada)
    newNode->data = data;
    newNode->next = nullptr;

    if (*head == nullptr) {
        *head = newNode;
    }
    else {
        Node* temp = *head;
        while (temp->next != nullptr) {
            temp = temp->next;
        }
        temp->next = newNode;
    }
}

// Versión en GPU de insertar un nodo en una posición específica
__global__ void insertAtPositionGPU(Node** head, int data, int position) {
    Node* newNode = (Node*)malloc(sizeof(Node));  // Asignación manual (no recomendada)
    newNode->data = data;

    if (position == 0) {
        newNode->next = *head;
        *head = newNode;
    }
    else {
        Node* temp = *head;
        int count = 0;
        while (temp != nullptr && count < position - 1) {
            temp = temp->next;
            count++;
        }

        if (temp != nullptr) {
            newNode->next = temp->next;
            temp->next = newNode;
        }
    }
}

// Versión en GPU de eliminar un nodo
__global__ void deleteNodeGPU(Node** head, int data) {
    if (*head == nullptr) return;

    Node* temp = *head;
    Node* prev = nullptr;

    if (temp != nullptr && temp->data == data) {
        *head = temp->next;
        free(temp);  // Usar free en lugar de delete
        return;
    }

    while (temp != nullptr && temp->data != data) {
        prev = temp;
        temp = temp->next;
    }

    if (temp == nullptr) return;

    prev->next = temp->next;
    free(temp);  // Usar free en lugar de delete
}

// Versión en GPU para imprimir la lista
__global__ void printListGPU(Node* head) {
    Node* current = head;
    while (current != nullptr) {
        printf("Data: %d\n", current->data);
        current = current->next;
    }
}

// Función para medir el tiempo de ejecución en la CPU
void measureCPUSpeed(Node** head) {
    auto start = std::chrono::high_resolution_clock::now();

    // Insertar nodos
    insertAtEndCPU(head, 10);
    insertAtEndCPU(head, 20);
    insertAtEndCPU(head, 30);

    // Imprimir lista
    printListCPU(*head);

    // Insertar en una posición específica
    insertAtPositionCPU(head, 25, 1);

    // Imprimir lista después de inserción
    printListCPU(*head);

    // Eliminar un nodo
    deleteNodeCPU(head, 20);

    // Imprimir lista después de eliminación
    printListCPU(*head);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Tiempo de ejecución en CPU: " << elapsed.count() << " segundos" << std::endl;
}

// Función para medir el tiempo de ejecución en la GPU
void measureGPUSpeed(Node** d_head) {
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Comienza la medición

    // Insertar nodos
    insertAtEndGPU << <1, 1 >> > (d_head, 10);
    insertAtEndGPU << <1, 1 >> > (d_head, 20);
    insertAtEndGPU << <1, 1 >> > (d_head, 30);
    cudaDeviceSynchronize();

    // Imprimir lista
    printListGPU << <1, 1 >> > (*d_head);
    cudaDeviceSynchronize();

    // Insertar en una posición específica
    insertAtPositionGPU << <1, 1 >> > (d_head, 25, 1);
    cudaDeviceSynchronize();

    // Imprimir lista después de inserción
    printListGPU << <1, 1 >> > (*d_head);
    cudaDeviceSynchronize();

    // Eliminar un nodo
    deleteNodeGPU << <1, 1 >> > (d_head, 20);
    cudaDeviceSynchronize();

    // Imprimir lista después de eliminación
    printListGPU << <1, 1 >> > (*d_head);
    cudaDeviceSynchronize();

    cudaEventRecord(stop); // Detiene la medición
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiempo de ejecución en GPU: " << elapsedTime / 1000.0 << " segundos" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    Node* headCPU = nullptr;
    Node** d_headGPU;
    cudaMallocManaged(&d_headGPU, sizeof(Node*));
    *d_headGPU = nullptr;

    std::cout << "Comparando tiempos de ejecución CPU vs GPU:\n";

    // Medir tiempo en CPU
    std::cout << "Versión en CPU:\n";
    measureCPUSpeed(&headCPU);

    // Medir tiempo en GPU
    std::cout << "Versión en GPU:\n";
    measureGPUSpeed(d_headGPU);

    // Liberar memoria
    cudaFree(d_headGPU);

    return 0;
}
