// DijkstraMPI_TCC.cpp : Este arquivo contém a função 'main'. A execução do programa começa e termina ali.
//

#define _CRT_SECURE_NO_WARNINGS
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>


#define MAX_NOS 16384
#define NUM_VERTICES 256 /*512*/ /*1024*/ /*2048*/ /*4096*/ /*8192*/ /*16384*/
#define MIN_PESO 1
#define MAX_PESO 20
#define DIST_MAX INT_MAX

struct No {
    int vertice;
    int peso;
    struct No* prox;
};

struct Grafo {
    struct No* cabeca[NUM_VERTICES];
    int numVertices;
};

struct No* criarNo(int v, int p) {
    struct No* novoNo = (struct No*)malloc(sizeof(struct No));
    novoNo->vertice = v;
    novoNo->peso = p;
    novoNo->prox = NULL;
    return novoNo;
}

struct Grafo* criarGrafo(int vertices) {
    struct Grafo* grafo = (struct Grafo*)malloc(sizeof(struct Grafo));
    grafo->numVertices = vertices;

    for (int i = 0; i < vertices; i++) {
        grafo->cabeca[i] = NULL;
    }

    return grafo;
}

void adicionarAresta(struct Grafo* grafo, int orig, int dest, int peso) {
    struct No* novoNo = criarNo(dest, peso);
    novoNo->prox = grafo->cabeca[orig];
    grafo->cabeca[orig] = novoNo;
}

void imprimirGrafo(struct Grafo* grafo) {
    printf("\nGrafo:\n");
    for (int i = 0; i < grafo->numVertices; i++) {
        struct No* temp = grafo->cabeca[i];
        printf("Vertice %d: ", i);
        while (temp != NULL) {
            printf("(%d,%d) -> ", temp->vertice, temp->peso);
            temp = temp->prox;
        }
        printf("NULL\n");
    }
}

// Função para salvar o grafo em um arquivo
void salvarGrafo(struct Grafo* grafo, const char* nomeArquivo) {
    FILE* arquivo = fopen(nomeArquivo, "w");
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo %s.\n", nomeArquivo);
        return;
    }

    // Escreve o número de vértices no arquivo
    fprintf(arquivo, "%d\n", grafo->numVertices);

    // Escreve as arestas do grafo no arquivo
    for (int i = 0; i < grafo->numVertices; i++) {
        struct No* temp = grafo->cabeca[i];
        while (temp != NULL) {
            fprintf(arquivo, "%d %d %d\n", i, temp->vertice, temp->peso);
            temp = temp->prox;
        }
    }

    fclose(arquivo);
    printf("Grafo salvo com sucesso no arquivo %s.\n", nomeArquivo);
}

// Função para carregar o grafo de um arquivo
struct Grafo* carregarGrafo(const char* nomeArquivo) {
    FILE* arquivo = fopen(nomeArquivo, "r");
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo %s.\n", nomeArquivo);
        return NULL;
    }

    int numVertices;
    fscanf(arquivo, "%d", &numVertices);

    struct Grafo* grafo = criarGrafo(numVertices);

    int origem, destino, peso;
    while (fscanf(arquivo, "%d %d %d", &origem, &destino, &peso) == 3) {
        adicionarAresta(grafo, origem, destino, peso);
    }

    fclose(arquivo);
    printf("Grafo carregado com sucesso do arquivo %s.\n", nomeArquivo);
    return grafo;
}

void dijkstraMPI(struct Grafo* grafo, int inicio) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int distancias[NUM_VERTICES];
    bool visitados[NUM_VERTICES];

    for (int i = 0; i < NUM_VERTICES; i++) {
        distancias[i] = INT_MAX;
        visitados[i] = false;
    }

    distancias[inicio] = 0;

    // Calcular o número de iterações para cada processo
    int iterations_per_process = (NUM_VERTICES - 1) / size;
    int remainder = (NUM_VERTICES - 1) % size;

    // Calcular os deslocamentos e tamanhos para MPI_Scatterv
    int displacements[256];
    int counts[256];
    for (int i = 0; i < size; i++) {
        counts[i] = iterations_per_process;
        if (i < remainder) {
            counts[i]++;
        }
        displacements[i] = i * iterations_per_process + (i < remainder ? i : remainder);
    }

    // Cada processo executa sua parte do laço externo
    for (int count = 0; count < iterations_per_process; count++) {
        int local_min_dist = INT_MAX;
        int local_min_vertex = -1;

        // Encontrar o vértice localmente não visitado com a menor distância
        for (int i = rank + count * size; i < NUM_VERTICES; i += size * iterations_per_process) {
            if (!visitados[i] && distancias[i] < local_min_dist) {
                local_min_dist = distancias[i];
                local_min_vertex = i;
            }
        }

        // Reduzir localmente para encontrar o mínimo global
        struct {
            int dist;
            int vertice;
        } local_min, global_min;

        local_min.dist = local_min_dist;
        local_min.vertice = local_min_vertex;

        MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        // Marcar o vértice globalmente mínimo como visitado
        if (global_min.vertice != -1) {
            visitados[global_min.vertice] = true;
        }

        // Atualizar as distâncias localmente
        struct No* v = grafo->cabeca[global_min.vertice];
        while (v != NULL) {
            if (!visitados[v->vertice] &&
                global_min.dist + v->peso < distancias[v->vertice]) {
                distancias[v->vertice] = global_min.dist + v->peso;
            }
            v = v->prox;
        }
    }

    // Atualizar as distâncias em cada processo usando MPI_Allreduce
    MPI_Allreduce(MPI_IN_PLACE, distancias, NUM_VERTICES, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    //Exibir as distâncias mínimas a partir do processo 0
    if (rank == 0) {
        printf("\nDistancias minimas a partir do vertice %d:\n", inicio);
        for (int i = 0; i < NUM_VERTICES; i++) {
            printf("Vertice %d: %d\n", i, distancias[i]);
        }
    }
}

void liberarGrafo(struct Grafo* grafo) {
    for (int i = 0; i < NUM_VERTICES; i++) {
        struct No* atual = grafo->cabeca[i];
        while (atual != NULL) {
            struct No* proximo = atual->prox;
            free(atual);
            atual = proximo;
        }
    }
    free(grafo);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*if (size!=2) {
        printf("Esta aplicacao requer 2 processos MPI.\n");
        MPI_Finalize();
        return 1;
    }      */

    int num_vertices = NUM_VERTICES / size; // Divide o número de vértices igualmente entre os processos
    int inicio = rank * num_vertices; // Define o início do intervalo para cada processo
    int fim = (rank + 1) * num_vertices; // Define o fim do intervalo para cada processo

    printf("\n***********************RANK - ID DO PROCESSO********** = %d\n", rank);
    printf("\n***********************SIZE - NUMERO DE THREADS******* = %d\n", size);
    printf("\n*******************NUMERO DE VERTICES***************** = %d\n", num_vertices);
    printf("\n***********************INICIO************************* = %d\n", inicio);
    printf("\n***********************FIM**************************** = %d\n", fim);

    struct Grafo* grafo = criarGrafo(NUM_VERTICES);
    int peso = 0;
    int numArestas = 0;
    int vertice_de_entrada = 0;

    const char* grafo256 = "D:\\Grafos\\grafo256.txt";
    //const char* grafo512 = "D:\\Grafos\\grafo512.txt";
    //const char* grafo1024 = "D:\\Grafos\\grafo1024.txt";
    //const char* grafo2048 = "D:\\Grafos\\grafo2048.txt";
    //const char* grafo4096 = "D:\\Grafos\\grafo4096.txt";
    //const char* grafo8192 = "D:\\Grafos\\grafo8192.txt";
    //const char* grafo16384 = "D:\\Grafos\\grafo16384.txt";

    /*for (int i = 0; i < NUM_VERTICES; i++) {
        for (int j = i + 1; j < NUM_VERTICES; j++) {
            peso++;
            adicionarAresta(grafo, i, j, peso);
            adicionarAresta(grafo, j, i, peso);

            numArestas++;
            if (peso > 20)
                peso = 0;
        }
    }*/

    //Cálculo do Tamanho do Grafo
    for (int i = 0; i < NUM_VERTICES; i++) {
        for (int j = i + 1; j < NUM_VERTICES; j++) {
            numArestas++;
        }
    }

    printf("Numero de Vertices = %d\n", NUM_VERTICES);
    printf("Numero de Arestas = %d\n", numArestas);

    grafo = carregarGrafo(grafo256);

    //imprimirGrafo(grafo);

    int finalizado = 0;
    MPI_Barrier(MPI_COMM_WORLD); // Sincroniza os processos antes de iniciar a medição de tempo
    double start_time = 0;
    double end_time = 0;

    for (int i = 0; i < 30; i++) {
        start_time = MPI_Wtime();
        dijkstraMPI(grafo, 0);
        end_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD); // Sincroniza os processos antes de exibir o tempo de execução 
        printf("Tempo de execucao %d = %3.5f ms\n", i + 1, (end_time - start_time) * 1000);
        //finalizado++;
    }

    liberarGrafo(grafo);
    MPI_Finalize();
    return 0;
}
