#!/usr/bin/env ruby

length = Integer(ARGV[0])
abort "Should really be a mutliple of 256" unless length % 256 == 0

vector_filepath = "v#{length}.dat"
vector_cubed_filepath = "v#{length}_cubed.dat"
vector_file = File.open vector_filepath, "w"
vector_cubed_file = File.open vector_cubed_filepath, "w"

random_generator = Random.new
random_range = -100.0..100.0

vector_file.write "#{length}\n"
vector_cubed_file.write "#{length}\n"
length.times do
  x = random_generator.rand(random_range)
  vector_file.write "#{x}\n"
  vector_cubed_file.write "#{x**3}\n"
end
