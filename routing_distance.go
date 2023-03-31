package main

import (
	"fmt"
	"log"
	"os"
	"github.com/RyanCarrier/dijkstra"
	"net"
	"strconv"
	"strings"
)


func main() {
	fmt.Println("started dijkstra graph")
	port := os.Args[1]
	//edges go path
	path := "data/output/edges_go.txt"
	graph, err := dijkstra.Import(path)
	fmt.Println("built dijkstra graph")
	if err!=nil{
                log.Fatal(err)
        }

	   port = fmt.Sprintf(":%s", port)
	   ln, err := net.Listen("tcp", port)
	   fmt.Println("listening to ", port)
        if err != nil {
                        // handle error
						fmt.Println("error ocurred")
        }

        for {
                conn, _ := ln.Accept()
                go handle(conn, graph, false)
        }


}

func handle(conn net.Conn, graph dijkstra.Graph, detail bool){
        buf := make([]byte, 16384)
        n,err := conn.Read(buf)
        data := string(buf[:n])
		fmt.Println("Raw received data:", data)
        x := strings.Split(data, ",")
        origin, err := strconv.Atoi(x[0])
        destination, err := strconv.Atoi(x[1])
        fmt.Println("received:", origin, destination)



	best, err := graph.Shortest(origin, destination)
	if err!=nil{
		log.Println(err)
		conn.Write([]byte("-1:[]"))
	} else {
		var distance string = ""
		var path string  = "[" + strconv.Itoa(best.Path[0])
		for i := 1; i < len(best.Path); i++{
			path =  path + "," + strconv.Itoa(best.Path[i]) 
		}
		path = path + "]"
		fmt.Println("Path:", path)
		distance = strconv.FormatInt(best.Distance, 10) 
		path = distance + ":" + path
		conn.Write([]byte(path))
	}
}
