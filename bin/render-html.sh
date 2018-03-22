#!/bin/bash

I="/Users/parrt/github/random-forest-importances/article/"
O="/tmp/rfimportances"

while true
do
	if test $I/article.md -nt $O/article.html
	then
		java -jar /Users/parrt/github/bookish/target/bookish-1.0-SNAPSHOT.jar -target html -o $O $I/article.json
	fi
	sleep .2s
done
