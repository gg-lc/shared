scaling:
	python -m grpc_tools.protoc --python_out=. --pyi_out=. --grpc_python_out=. -I. scaling.proto
service:
	python -m grpc_tools.protoc --python_out=. --pyi_out=. --grpc_python_out=. -I. service.proto
test:
	python -m grpc_tools.protoc --python_out=. --pyi_out=. --grpc_python_out=. -I. test.proto
clean:
	rm *.py *.pyi
